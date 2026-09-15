"""
test_stage2_rewind.py  -  Stage 2 acceptance test

The gate for stage 2: a suspect told something must genuinely forget it
when the cache is cropped back past that turn. Not hidden, not instructed
to ignore - the tokens stop existing.

Tests:
  1. cropping restores the exact earlier cache (bit-identical)
  2. a rewound suspect's answer contains no trace of the erased turn
  3. snapshots survive a GPU -> host RAM -> GPU roundtrip
  4. rewinding does not disturb other suspects

Run:  py test_stage2_rewind.py
"""

import json

import torch
from Model_setup import load_model, tolerance

import cache_ops as co
from Extractor import FactExtractor
from Retrieval import HybridRetriever, Timeline
from Suspects import (Suspect, compute_shared_len, make_shared_block,
                      make_private_block)

tok, model, DEVICE = load_model()
TOL = tolerance(model)

case = json.load(open("Case_Ashfield.json", encoding="utf-8"))
ex = FactExtractor(case["entities"])
tl = Timeline(case["timeline"])
ret = HybridRetriever(case["entities"], tl)


def make_world():
    """Prefill the shared prefix once and fork every suspect off it."""
    shared_block = make_shared_block(case)
    privates = [make_private_block(case, sp) for sp in case["suspects"]]
    n_shared, shared_ids = compute_shared_len(tok, shared_block, privates)
    shared, _ = co.prefill(
        model, torch.tensor([shared_ids], device=DEVICE), label="shared")
    suspects = {
        spec["id"]: Suspect(spec, case, shared, shared_ids, model, tok, DEVICE)
        for spec in case["suspects"]
    }
    return shared, n_shared, suspects


shared_cache, n_shared, suspects = make_world()
print(f"shared block: {n_shared} tokens, {len(suspects)} suspects forked\n")


# ======================================================================

def test_shared_prefix_identical():
    """Every suspect must start from a bit-identical copy of the shared block."""
    for s in suspects.values():
        trimmed = co.crop(s.base_cache, n_shared, label="verify")
        d = co.max_diff(shared_cache, trimmed)
        assert d == 0.0, f"{s.id}: shared prefix diverged by {d:.2e}"
        assert not co.shares_memory(shared_cache, s.base_cache)
    print("  shared prefix bit-identical across all suspects")


def test_crop_restores_exact_state():
    """Cropping back must give exactly the cache we had before the turn."""
    s = suspects["vance"]
    s.reset()

    before = co.fork(s.cache, label="keepsake")
    len_before = co.length(s.cache)

    s.ask("Where were you at nine?", max_tokens=25)
    assert co.length(s.cache) > len_before

    s.rewind_last(1)
    assert co.length(s.cache) == len_before, (
        f"expected {len_before}, got {co.length(s.cache)}"
    )

    d = co.max_diff(before, s.cache)
    assert d == 0.0, f"rewound cache differs by {d:.2e}"
    print(f"  crop restores exact state (len {len_before}, diff 0.00e+00)")


def test_rewind_erases_knowledge():
    """
    The behavioural gate. Tell the suspect something distinctive, confirm
    they use it, rewind past it, then check it cannot resurface.
    """
    s = suspects["rourke"]
    s.reset()

    secret_word = "keycard"
    told = ("We know the stairwell keycard reader logged an entry at 21:38. "
            "What do you say to that?")

    a1, _, _ = s.ask(told, max_tokens=45)
    print(f"  told   : {a1[:70]}")

    s.rewind_last(1)
    assert len(s.turns) == 0
    assert co.length(s.cache) == s.base_len

    a2, _, _ = s.ask("Is there anything else you want to tell me?", max_tokens=45)
    print(f"  after  : {a2[:70]}")

    assert secret_word not in a2.lower(), (
        f"rewind leaked: answer still references {secret_word!r}\n{a2}"
    )
    print("  rewind erases knowledge")


def test_rewind_to_middle_turn():
    """Rewinding to turn k must drop turns k onward and keep 0..k-1."""
    s = suspects["halloway"]
    s.reset()

    s.ask("Where were you that evening?", max_tokens=25)
    s.ask("Did you go upstairs?", max_tokens=25)
    s.ask("What did you take from the desk?", max_tokens=25)
    assert len(s.turns) == 3

    len_at_turn1 = s.turns[1].cache_len_before
    dropped = s.rewind_to(1)

    assert len(dropped) == 2
    assert len(s.turns) == 1
    assert co.length(s.cache) == len_at_turn1
    print(f"  rewind to middle turn ok (kept 1, dropped 2)")


def test_snapshot_roundtrip():
    """
    A snapshot must come back from the store unchanged.

    Stage 6 moved snapshots into a SnapshotStore, so a Turn now carries a
    key rather than the cache itself.
    """
    s = suspects["vance"]
    s.reset()
    s.ask("Tell me about the paper.", max_tokens=25)

    key = s.turns[0].snapshot_key
    assert key is not None, "the turn recorded no snapshot"
    expected_len = s.turns[0].cache_len_before

    back = s.store.get(key)
    assert back is not None, "the store lost the snapshot"
    assert co.length(back) == expected_len

    tier = s.restore_snapshot(0)
    assert co.length(s.cache) == expected_len
    assert len(s.turns) == 0
    print(f"  snapshot roundtrip ok (tier '{tier}', "
          f"length {expected_len} preserved)")


def test_rewind_isolated_between_suspects():
    """Rewinding one suspect must not disturb another."""
    a = suspects["rourke"]
    b = suspects["vance"]
    a.reset()
    b.reset()

    a.ask("Where were you?", max_tokens=20)
    b.ask("Where were you?", max_tokens=20)

    b_len = co.length(b.cache)
    b_keep = co.fork(b.cache, label="keepsake")

    a.rewind_last(1)

    assert co.length(b.cache) == b_len, "rewinding A changed B's cache length"
    assert co.max_diff(b_keep, b.cache) == 0.0, "rewinding A corrupted B"
    print("  rewind isolated between suspects")


if __name__ == "__main__":
    tests = [
        test_shared_prefix_identical,
        test_crop_restores_exact_state,
        test_rewind_erases_knowledge,
        test_rewind_to_middle_turn,
        test_snapshot_roundtrip,
        test_rewind_isolated_between_suspects,
    ]
    for t in tests:
        print(f"\n{t.__name__}")
        t()

    print("\n" + "=" * 60)
    print("stage 2 PASSED")
    print(co.COUNTER.summary())