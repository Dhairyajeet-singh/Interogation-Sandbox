"""
test_cache_ops.py
-----------------
The stage 0 experiments, as assertions.

Run with:  python test_cache_ops.py
(or pytest, if you prefer)

These exist so that when stage 5 starts forking caches in anger, a broken
fork fails loudly here instead of showing up as one suspect mysteriously
knowing another's secret.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import cache_ops as co

NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("loading model...")
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(
    NAME, torch_dtype=torch.float16, attn_implementation="eager"
).to("cuda")
model.eval()

SYSTEM = (
    "You are a witness. Case file: Elena Marsh died in the observatory "
    "between 9pm and 10pm. The murder weapon was a brass counterweight. "
    "Rain began at 9:20pm. A wet umbrella was found in the east hall. "
    "Answer using only these facts, in one short sentence."
)
Q1 = "What was the murder weapon?"
Q2 = "When did the rain start?"


def max_diff(a, b):
    """Largest absolute difference across layer-0 keys."""
    return (a[0][0].float() - b[0][0].float()).abs().max().item()


def test_shapes():
    ids, n_shared = co.split_ids(tok, SYSTEM, Q1)
    cache, _ = co.prefill(model, ids[:, :n_shared])

    assert co.length(cache) == n_shared
    assert co.n_layers(cache) == model.config.num_hidden_layers
    assert cache[0][0].shape[1] == model.config.num_key_value_heads

    bpt = co.bytes_per_token(cache)
    expected = (2 * model.config.num_hidden_layers
                * model.config.num_key_value_heads
                * (model.config.hidden_size // model.config.num_attention_heads)
                * 2)  # fp16
    assert abs(bpt - expected) < 1, f"{bpt} != {expected}"
    print(f"  shapes ok - {co.n_layers(cache)} layers, {bpt:.0f} B/token")


def test_crop_is_exact():
    """
    Causal attention means later tokens never influenced earlier ones,
    so cropping is bit-identical to a fresh prefill of the same prefix.
    """
    ids, n_shared = co.split_ids(tok, SYSTEM, Q1)

    full, _ = co.prefill(model, ids)
    cropped = co.crop(full, n_shared)
    fresh, _ = co.prefill(model, ids[:, :n_shared])

    d = max_diff(cropped, fresh)
    assert d == 0.0, f"crop should be exact, got {d:.2e}"
    print(f"  crop exact - diff {d:.2e}")


def test_fork_is_exact():
    """A forked cache extended with a question must equal a full prefill."""
    ids_a, n_shared = co.split_ids(tok, SYSTEM, Q1)

    shared, _ = co.prefill(model, ids_a[:, :n_shared])
    forked, _ = co.prefill(model, ids_a[:, n_shared:], co.fork(shared))
    fresh, _ = co.prefill(model, ids_a)

    d = max_diff(forked, fresh)
    assert d == 0.0, f"fork should be exact, got {d:.2e}"
    print(f"  fork exact - diff {d:.2e}")


def test_forks_are_independent():
    """Two forks must not share storage, or one suspect corrupts the other."""
    ids, n_shared = co.split_ids(tok, SYSTEM, Q1)
    shared, _ = co.prefill(model, ids[:, :n_shared])

    a, b = co.fork(shared), co.fork(shared)
    assert not co.shares_memory(a, b)
    assert not co.shares_memory(a, shared)

    # write into a and confirm the source is untouched
    before = shared[0][0][0, 0, 0, 0].item()
    a[0][0][0, 0, 0, 0] = 999.0
    assert shared[0][0][0, 0, 0, 0].item() == before
    print("  forks independent")


def test_crop_clones_not_views():
    """crop() must clone. A view would share memory with its source."""
    ids, n_shared = co.split_ids(tok, SYSTEM, Q1)
    full, _ = co.prefill(model, ids)

    cropped = co.crop(full, 4)
    assert not co.shares_memory(cropped, full)

    view = tuple((k[:, :, :4, :], v[:, :, :4, :]) for k, v in full)
    assert co.shares_memory(view, full), "sanity: a raw slice DOES share"
    print("  crop clones (raw slice would not)")


def test_snapshot_roundtrip():
    """A cache moved to host RAM and back must be unchanged."""
    ids, n_shared = co.split_ids(tok, SYSTEM, Q1)
    cache, _ = co.prefill(model, ids[:, :n_shared])

    snap = co.snapshot(cache, device="cpu")
    assert snap[0][0].device.type == "cpu"

    back = co.restore(snap, device="cuda")
    d = max_diff(cache, back)
    assert d == 0.0, f"roundtrip changed the cache: {d:.2e}"
    print(f"  snapshot roundtrip exact - diff {d:.2e}")


def test_rewind_erases_knowledge():
    """
    The behavioural test: a suspect told something must forget it
    completely when the cache is cropped back past that point.
    """
    ids, n_shared = co.split_ids(tok, SYSTEM, Q1)
    shared, _ = co.prefill(model, ids[:, :n_shared])

    told, logits = co.prefill(
        model, ids[:, n_shared:], co.fork(shared), label="q1"
    )
    ans1, _ = co.decode(model, tok, told, logits)

    rewound = co.crop(told, n_shared, label="rewind")
    assert co.length(rewound) == n_shared

    ids_b, n_b = co.split_ids(tok, SYSTEM, Q2)
    assert n_b == n_shared, "shared prefix must tokenise identically"

    after, logits2 = co.prefill(model, ids_b[:, n_b:], rewound, label="q2")
    ans2, _ = co.decode(model, tok, after, logits2)

    print(f"  before rewind: {ans1.strip()[:60]}")
    print(f"  after  rewind: {ans2.strip()[:60]}")
    assert "counterweight" not in ans2.lower(), \
        "rewind failed: answer still references the erased question"
    print("  rewind erases knowledge")


if __name__ == "__main__":
    tests = [
        test_shapes,
        test_crop_is_exact,
        test_fork_is_exact,
        test_forks_are_independent,
        test_crop_clones_not_views,
        test_snapshot_roundtrip,
        test_rewind_erases_knowledge,
    ]
    for t in tests:
        print(f"\n{t.__name__}")
        t()

    print("\n" + "=" * 50)
    print("all tests passed")
    print(co.COUNTER.summary())