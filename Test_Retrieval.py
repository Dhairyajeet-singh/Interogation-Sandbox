"""
test_retrieval.py  -  Stage 3 acceptance test

Checks that:
  - each search method finds the obvious thing
  - fusion combines them sensibly
  - the knowledge filter is absolute (this is the stage 3 gate)
  - the timeline graph adjudicates alibis correctly
"""

import json

from Extractor import FactExtractor
from Retrieval import HybridRetriever, Timeline, KnowledgeChecker, HAVE_EMBEDDINGS

case = json.load(open("case_ashfield.json", encoding="utf-8"))
ents = case["entities"]
ex = FactExtractor(ents)
tl = Timeline(case["timeline"])
ret = HybridRetriever(ents, tl)
kc = KnowledgeChecker(ex, case["public_facts"])

SUS = {s["id"]: s for s in case["suspects"]}


def test_keyword():
    hits = ret.search_keyword("what was the murder weapon")
    assert "ev_counterweight" in hits, hits
    hits = ret.search_keyword("the keycard reader on the stairwell")
    assert "ev_keycard" in hits, hits
    print("  keyword search ok")


def test_meaning():
    if not HAVE_EMBEDDINGS:
        print("  meaning search SKIPPED (sentence-transformers not installed)")
        return
    hits = ret.search_meaning("was anything taken from the room")
    assert ("ev_notebook" in hits or "ev_dust_outline" in hits), hits
    print("  meaning search ok")


def test_graph():
    hits = ret.search_graph("where was Vance at the time")
    assert "per_vance" in hits, hits
    assert "loc_observatory" in hits or "loc_study" in hits, hits
    print("  graph search ok")


def test_fusion_orders_sensibly():
    fused = ret.fuse([
        ["a", "b", "c"],
        ["b", "a", "d"],
        ["b", "x", "y"],
    ])
    assert fused[0] == "b", fused
    print("  RRF fusion ok")


def test_knowledge_filter_is_absolute():
    """
    The stage 3 gate. Rourke does not know about the keycard reader,
    so retrieval must never hand it to him no matter what is asked.
    """
    rourke = SUS["rourke"]
    for q in [
        "tell me about the keycard reader",
        "what does the stairwell log show",
        "the keycard, explain it",
    ]:
        got = ret.retrieve(q, rourke["knows"])
        assert "ev_keycard" not in got, f"leaked keycard to Rourke on {q!r}: {got}"

    vance = SUS["vance"]
    got = ret.retrieve("what happened with the paper", vance["knows"])
    assert "ev_paper" in got, got
    print("  knowledge filter absolute ok")


def test_violation_detection():
    rourke = SUS["rourke"]

    bad = "I saw the keycard log showing someone went up at 21:38."
    v = kc.violations(bad, rourke["knows"])
    assert "ev_keycard" in v, v

    ok = "I was in the greenhouse. The rain started at 9:20."
    v = kc.violations(ok, rourke["knows"])
    assert v == set(), f"false positive: {v}"

    # public facts are allowed to everyone
    v = kc.violations("She was hit with the brass counterweight.", rourke["knows"])
    assert v == set(), f"public fact flagged: {v}"
    print("  violation detection ok")


def test_alibi_checking():
    verdict, why = tl.check_alibi("per_vance", "loc_observatory", "21:40")
    assert verdict == "supported", (verdict, why)

    verdict, why = tl.check_alibi("per_vance", "loc_observatory", "21:10")
    assert verdict == "contradicted", (verdict, why)

    verdict, why = tl.check_alibi("per_rourke", "loc_greenhouse", "21:30")
    assert verdict == "contradicted", (verdict, why)
    print(f"  alibi checking ok  ({why})")


def test_who_was_where():
    at_obs = tl.who_was_at("loc_observatory", at="21:40")
    actors = {e["actor"] for e in at_obs}
    assert "per_vance" in actors and "per_marsh" in actors, actors

    at_obs = tl.who_was_at("loc_observatory", at="21:55")
    actors = {e["actor"] for e in at_obs}
    assert "per_vance" not in actors, actors
    print("  timeline lookup ok")


if __name__ == "__main__":
    print(f"embeddings available: {HAVE_EMBEDDINGS}\n")
    for fn in [
        test_keyword, test_meaning, test_graph, test_fusion_orders_sensibly,
        test_knowledge_filter_is_absolute, test_violation_detection,
        test_alibi_checking, test_who_was_where,
    ]:
        fn()
    print("\nstage 3 PASSED")