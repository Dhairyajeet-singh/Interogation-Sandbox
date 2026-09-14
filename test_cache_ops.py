"""
test_cache_ops.py   -  Stage 0 acceptance test
----------------------------------------------
The stage 0 experiments, as assertions, against the real model.

Two classes of check, and the distinction matters:

  EXACT      pure tensor operations - fork, crop, snapshot. These copy
             or slice numbers that already exist, so they must be
             bit-identical. Asserted == 0.0.

  TOLERANT   comparisons where the model RECOMPUTES something. Prefilling
             7 tokens onto 14 cached is a different sequence of floating
             point reductions than prefilling 21 at once, so the result
             can differ by ~1e-7 in fp32. That is arithmetic, not a bug.

Run:  py test_cache_ops.py
"""

import torch

import cache_ops as co
from Model_setup import load_model, tolerance

tok, model, DEVICE = load_model()
TOL = tolerance(model)
print(f"device={DEVICE} dtype={next(model.parameters()).dtype} tol={TOL}\n")

TEXT = ("Case file. Elena Marsh died in the observatory between 9pm and 10pm. "
        "The murder weapon was a brass counterweight. Rain began at 9:20pm.")
MORE = " A wet umbrella was found in the east hall."


def ids_of(text):
    return tok(text, return_tensors="pt",
               add_special_tokens=False).input_ids.to(DEVICE)


# ======================================================================
# shape and size
# ======================================================================

def test_shapes():
    ids = ids_of(TEXT)
    cache, _ = co.prefill(model, ids)

    assert isinstance(cache, tuple), f"normalise returned {type(cache).__name__}"
    assert co.length(cache) == ids.shape[1]
    assert co.n_layers(cache) == model.config.num_hidden_layers
    assert cache[0][0].shape[1] == model.config.num_key_value_heads

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    nbytes = cache[0][0].element_size()
    expected = (2 * model.config.num_hidden_layers
                * model.config.num_key_value_heads * head_dim * nbytes)
    got = co.bytes_per_token(cache)
    assert abs(got - expected) < 1, f"{got} != {expected}"
    print(f"  {co.n_layers(cache)} layers, "
          f"{model.config.num_key_value_heads} kv heads, "
          f"{got:.0f} B/token")


def test_cache_grows_by_one_per_decoded_token():
    ids = ids_of(TEXT)
    cache, logits = co.prefill(model, ids)
    before = co.length(cache)

    text, cache, produced = co.decode(model, tok, cache, logits, max_tokens=8)

    assert co.length(cache) == before + len(produced), (
        f"{before} + {len(produced)} != {co.length(cache)}")
    print(f"  decoded {len(produced)} tokens, cache {before} -> {co.length(cache)}")


# ======================================================================
# EXACT - pure tensor operations
# ======================================================================

def test_fork_is_an_independent_copy():
    cache, _ = co.prefill(model, ids_of(TEXT))
    a, b = co.fork(cache), co.fork(cache)

    assert co.max_diff(cache, a) == 0.0
    assert not co.shares_memory(a, b)
    assert not co.shares_memory(a, cache)

    before = cache[0][0][0, 0, 0, 0].item()
    a[0][0][0, 0, 0, 0] = 999.0
    assert cache[0][0][0, 0, 0, 0].item() == before, "fork shared storage"
    print("  fork exact and independent")


def test_crop_clones_not_views():
    cache, _ = co.prefill(model, ids_of(TEXT))

    cropped = co.crop(cache, 5)
    assert co.length(cropped) == 5
    assert not co.shares_memory(cropped, cache)

    view = tuple((k[:, :, :5, :], v[:, :, :5, :]) for k, v in cache)
    assert co.shares_memory(view, cache), "sanity: a raw slice DOES share"
    assert co.max_diff(cropped, view) == 0.0
    print("  crop clones; a raw slice would not")


def test_snapshot_roundtrip_is_exact():
    cache, _ = co.prefill(model, ids_of(TEXT))

    snap = co.snapshot(cache, device="cpu")
    assert snap[0][0].device.type == "cpu"

    back = co.restore(snap, device=DEVICE)
    assert co.max_diff(cache, back) == 0.0
    print("  snapshot gpu -> cpu -> gpu exact")


# ======================================================================
# TOLERANT - the model recomputes
# ======================================================================

def test_crop_matches_a_fresh_prefill():
    """
    Attention is causal: tokens after position n never influenced tokens
    before it. So cropping a long cache gives the same numbers as
    prefilling only the prefix.
    """
    ids = ids_of(TEXT)
    n = 8

    full, _ = co.prefill(model, ids)
    cropped = co.crop(full, n)
    fresh, _ = co.prefill(model, ids[:, :n])

    d = co.max_diff(cropped, fresh)
    assert d <= TOL, f"crop diverged from fresh prefill by {d:.2e}"
    print(f"  crop vs fresh prefill: {d:.2e}")


def test_fork_then_extend_matches_full_prefill():
    """
    The claim the whole project rests on: prefill a prefix, fork it,
    extend the fork - and get what a single full prefill would have given.
    """
    a = ids_of(TEXT)
    b = ids_of(MORE)

    shared, _ = co.prefill(model, a)
    extended, _ = co.prefill(model, b, co.fork(shared))
    full, _ = co.prefill(model, torch.cat([a, b], dim=1))

    assert co.length(extended) == co.length(full)
    d = co.max_diff(extended, full)
    assert d <= TOL, f"fork+extend diverged from full prefill by {d:.2e}"
    print(f"  fork+extend vs full prefill: {d:.2e}")


def test_two_forks_diverge_independently():
    """Two forks given different continuations must not affect each other."""
    base, _ = co.prefill(model, ids_of(TEXT))

    a, _ = co.prefill(model, ids_of(" Who saw you there?"), co.fork(base))
    b, _ = co.prefill(model, ids_of(" Where were you at nine?"), co.fork(base))

    assert co.max_diff(co.crop(a, co.length(base)), base) <= TOL
    assert co.max_diff(co.crop(b, co.length(base)), base) <= TOL

    n = min(co.length(a), co.length(b))
    tail_a = tuple((k[:, :, co.length(base):n, :],
                    v[:, :, co.length(base):n, :]) for k, v in a)
    tail_b = tuple((k[:, :, co.length(base):n, :],
                    v[:, :, co.length(base):n, :]) for k, v in b)
    assert co.max_diff(tail_a, tail_b) > TOL, "different questions gave identical KV"
    print("  forks share their prefix and diverge after it")


# ======================================================================

if __name__ == "__main__":
    tests = [
        test_shapes,
        test_cache_grows_by_one_per_decoded_token,
        test_fork_is_an_independent_copy,
        test_crop_clones_not_views,
        test_snapshot_roundtrip_is_exact,
        test_crop_matches_a_fresh_prefill,
        test_fork_then_extend_matches_full_prefill,
        test_two_forks_diverge_independently,
    ]
    for t in tests:
        print(f"\n{t.__name__}")
        t()

    print("\n" + "=" * 60)
    print("stage 0 PASSED")
    print(co.COUNTER.summary())