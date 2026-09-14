"""
test_stage5_Branching.py  -  Stage 5 acceptance test

Two things must hold:
  1. three forks cost near-zero prefill - the shared block is not re-read
  2. scores are stable - run explore() twice on the same state and get
     the same numbers (greedy decoding, no sampling)

Plus: the committed answer is the answer the player was shown, and the
discarded forks leave no trace on the suspect.

Run:  py test_stage5_Branching.py
"""

import json

import torch
from Model_setup import load_model, tolerance

import Branching
import cache_ops as co
from Extractor import FactExtractor
from Retrieval import HybridRetriever, Timeline
from Suspects import (Suspect, compute_shared_len, make_shared_block,
                      make_private_block)

tok, model, DEVICE = load_model()
TOL = tolerance(model)

case = json.load(open("case_ashfield.json", encoding="utf-8"))
ex = FactExtractor(case["entities"])
tl = Timeline(case["timeline"])
ret = HybridRetriever(case["entities"], tl)

shared_block = make_shared_block(case)
privates = [make_private_block(case, sp) for sp in case["suspects"]]
n_shared, shared_ids = compute_shared_len(tok, shared_block, privates)
shared, _ = co.prefill(
    model, torch.tensor([shared_ids], device=DEVICE), label="shared")
suspects = {
    spec["id"]: Suspect(spec, case, shared, shared_ids, model, tok, DEVICE)
    for spec in case["suspects"]
}
print(f"shared block: {n_shared} tokens\n")


def test_forks_are_cheap():
    """
    Three trial questions must cost only their own tokens, not three
    re-reads of the shared block. This is the whole justification for
    the feature existing.
    """
    s = suspects["vance"]
    s.reset()
    s.ask("Where were you at nine?", max_tokens=25)

    cands, cost = Branching.explore(s, case, tl, ex, ret, n=3)

    assert len(cands) == 3, f"expected 3 candidates, got {len(cands)}"
    assert cost["forks"] == 3, cost

    # what three naive trials would have cost
    naive = 3 * (co.length(s.cache) + 20)
    print(f"  3 forks cost {cost['prefill_tokens']} prefill tokens")
    print(f"  naive would cost ~{naive} tokens "
          f"({100*(1 - cost['prefill_tokens']/naive):.0f}% cheaper)")
    assert cost["prefill_tokens"] < naive / 3, (
        f"forks cost {cost['prefill_tokens']}, expected well under {naive/3:.0f}"
    )


def test_scores_are_reproducible():
    """Run explore twice on the same state; the numbers must match."""
    s = suspects["halloway"]
    s.reset()
    s.ask("Where were you that evening?", max_tokens=25)

    first, _ = Branching.explore(s, case, tl, ex, ret, n=3)
    second, _ = Branching.explore(s, case, tl, ex, ret, n=3)

    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.question == b.question, (a.question, b.question)
        assert a.answer == b.answer, f"\n{a.answer!r}\n{b.answer!r}"
        assert abs(a.score - b.score) < 1e-9, (a.score, b.score)
    print(f"  scores stable across runs "
          f"({[round(c.score, 3) for c in first]})")


def test_exploring_leaves_no_trace():
    """Discarded forks must not touch the suspect's live state."""
    s = suspects["rourke"]
    s.reset()
    s.ask("Where were you?", max_tokens=25)

    len_before = co.length(s.cache)
    turns_before = len(s.turns)
    keepsake = co.fork(s.cache, label="keepsake")

    Branching.explore(s, case, tl, ex, ret, n=3)

    assert co.length(s.cache) == len_before, "exploring advanced the cache"
    assert len(s.turns) == turns_before, "exploring recorded a turn"
    assert co.max_diff(keepsake, s.cache) == 0.0, "exploring mutated the cache"
    print("  exploring leaves the suspect untouched")


def test_commit_uses_the_previewed_answer():
    """
    The score the player saw described a specific answer. Committing must
    adopt that exact fork, not regenerate - otherwise the number was a
    promise about text they never receive.
    """
    s = suspects["vance"]
    s.reset()
    s.ask("Where were you at nine?", max_tokens=25)

    cands, _ = Branching.explore(s, case, tl, ex, ret, n=3)
    chosen = cands[0]

    answer = Branching.commit(s, chosen, ret, case)

    assert answer == chosen.answer, "committed answer differs from the preview"
    assert s.turns[-1].answer == chosen.answer
    assert co.length(s.cache) == co.length(chosen.cache)
    print("  committed answer matches the preview exactly")


def test_scoring_components_are_counted():
    """Scores must come from counted entities, never from a model."""
    s = suspects["halloway"]
    s.reset()

    c = Branching.Candidate(
        kind="test",
        question="q",
        answer="I was in the east hall at 21:50 and called straight away.",
    )
    Branching.score_candidate(c, s, case, tl, ex)

    assert c.coverage >= 2, (c.coverage, c.new_entities)
    assert c.verifiable >= 2, c.verifiable
    assert 0.0 <= c.score <= 1.0, c.score

    empty = Branching.Candidate(kind="t", question="q",
                                answer="I don't remember anything.")
    Branching.score_candidate(empty, s, case, tl, ex)
    assert empty.coverage == 0, empty.new_entities
    assert empty.score < c.score
    print(f"  scoring counted correctly "
          f"(informative {c.score:.2f} > evasive {empty.score:.2f})")


def test_candidates_respect_knowledge():
    """Retrieved facts handed to a candidate must be inside the knows-list."""
    s = suspects["rourke"]
    s.reset()
    allowed = set(s.knows) | set(case["public_facts"])

    cands, _ = Branching.explore(s, case, tl, ex, ret, n=3)
    for c in cands:
        hits = ret.retrieve(c.question, list(allowed), top_k=3)
        assert all(h in allowed for h in hits), (c.question, hits)
    print("  candidate retrieval respects the knowledge filter")


if __name__ == "__main__":
    tests = [
        test_forks_are_cheap,
        test_scores_are_reproducible,
        test_exploring_leaves_no_trace,
        test_commit_uses_the_previewed_answer,
        test_scoring_components_are_counted,
        test_candidates_respect_knowledge,
    ]
    for t in tests:
        print(f"\n{t.__name__}")
        t()

    print("\n" + "=" * 60)
    print("stage 5 PASSED")
    print(co.COUNTER.summary())