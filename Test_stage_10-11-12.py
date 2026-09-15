"""
Test_Stages101112.py
--------------------
Acceptance tests for:

  Stage 10  case generation and the validator
  Stage 11  the judge
  Stage 12  the experiment harness

Runs with no API key. The DeepSeek path is exercised as far as it can be
without one: the client is built, the request is shaped, and the offline
fallback takes over. Set DEEPSEEK_API_KEY and the same tests exercise the
real model instead - nothing else changes.

Run:  py Test_Stages101112.py
"""

import json
import os
from pathlib import Path

import torch

import Branching
import cache_ops as co
import Experiments
import Interrogation
from Deepseek import Deepseek, DeepseekError, strip_json, have_key
from Extractor import FactExtractor
from Generator import CaseGenerator, offline_case, validate
from Judge import CaseJudge, build_transcript, truth_summary, mechanical_verdict
from Model_setup import load_model
from Retrieval import HybridRetriever, Timeline
from Suspects import (Suspect, compute_shared_len, make_shared_block,
                      make_private_block)

HERE = Path(__file__).parent
ONLINE = have_key()


# ======================================================================
# STAGE 10
# ======================================================================

def test_validator_accepts_good_cases():
    """The hand-written case and every offline size must pass."""
    ash = json.loads((HERE / "Case_Ashfield.json").read_text(encoding="utf-8"))
    errs = validate(ash)
    assert not errs, f"Case_Ashfield.json is invalid: {errs}"

    for n in range(2, 9):
        c = offline_case(n, seed=n)
        e = validate(c)
        assert not e, f"offline case with {n} suspects invalid: {e}"
    print("  hand-written case and offline sizes 2-8 all valid")


def test_validator_catches_each_kind_of_break():
    """
    Every rule the validator claims to enforce, broken one at a time.
    A validator nobody has tried to fool is not a validator.
    """
    def broken(mutate):
        c = json.loads(json.dumps(offline_case(3, seed=1)))
        mutate(c)
        return validate(c)

    def has(errors, fragment):
        return any(fragment in e for e in errors)

    # culprit is not a suspect
    e = broken(lambda c: c.__setitem__("culprit", "nobody"))
    assert has(e, "not one of the suspects"), e

    # culprit has no opportunity
    def no_opportunity(c):
        c["timeline"] = [t for t in c["timeline"]
                         if not (t["actor"] == f"per_{c['culprit']}"
                                 and t["place"] == c["crime_scene"])]
    e = broken(no_opportunity)
    assert has(e, "no opportunity"), e

    # two places at once
    def bilocate(c):
        a = f"per_{c['culprit']}"
        c["timeline"].append({"actor": a, "place": "loc_cellar",
                              "from": "20:20", "to": "20:40",
                              "verifiable_by": None})
    e = broken(bilocate)
    assert has(e, "two places at once"), e

    # backwards time range
    e = broken(lambda c: c["timeline"][0].update({"from": "21:00", "to": "20:00"}))
    assert has(e, "runs backwards"), e

    # malformed clock
    e = broken(lambda c: c["timeline"][0].update({"from": "25:99"}))
    assert has(e, "24-hour HH:MM"), e

    # too few aliases
    def strip_aliases(c):
        first = next(iter(c["entities"]))
        c["entities"][first]["aliases"] = ["only one"]
    e = broken(strip_aliases)
    assert has(e, "aliases"), e

    # ambiguous alias across two entities
    def collide(c):
        ids = [k for k in c["entities"] if k.startswith("loc_")][:2]
        c["entities"][ids[1]]["aliases"] = list(
            c["entities"][ids[0]]["aliases"])
    e = broken(collide)
    assert has(e, "claimed by both"), e

    # unreachable lie
    def unreachable(c):
        s = c["suspects"][0]
        s["lies"][0]["exposed_by"] = list(s["knows"])[:1]
    e = broken(unreachable)
    assert has(e, "already know"), e

    # a suspect with no lies
    e = broken(lambda c: c["suspects"][0].__setitem__("lies", []))
    assert has(e, "no lies"), e

    # dangling reference
    e = broken(lambda c: c["suspects"][0]["knows"].append("ev_does_not_exist"))
    assert has(e, "not an entity"), e

    # guilty flag out of step with culprit
    e = broken(lambda c: [s.__setitem__("guilty", True) for s in c["suspects"]])
    assert has(e, "exactly one suspect"), e

    print("  11 distinct breakages all caught")


def test_generator_offline_batch():
    gen = CaseGenerator(Deepseek(offline=True, verbose=False), verbose=False)
    out = HERE / "cases_test"
    written, summary = gen.batch(n=3, suspects=(2, 3, 4), out_dir=out)

    assert summary["written"] == 3, summary
    for p in written:
        c = json.loads(p.read_text(encoding="utf-8"))
        assert not validate(c)
        assert c["generated_by"] == "offline"
        FactExtractor(c["entities"])            # must build
    print(f"  wrote {summary['written']} valid cases, "
          f"first-try rate {summary['first_try_rate']:.0%}")

    for p in written:
        p.unlink()
    out.rmdir()


def test_deepseek_client_shape():
    """
    Without a key: refuses to pretend, and the fallback fires.
    With a key: a real call is made and parsed.
    """
    off = Deepseek(offline=True, verbose=False)
    assert off.offline
    try:
        off.chat("s", "u")
        raise AssertionError("offline client should refuse to call")
    except DeepseekError:
        pass

    data, source = off.chat_json("s", "u", fallback=lambda: {"ok": True})
    assert data == {"ok": True} and source == "offline"

    # the JSON extractor has to survive fenced and chatty replies
    assert json.loads(strip_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(strip_json('Sure! {"a": [1,2]} hope that helps'))["a"] == [1, 2]

    if ONLINE:
        live = Deepseek(verbose=False)
        out = live.chat("Reply with JSON only.",
                        'Return {"pong": true} and nothing else.')
        assert json.loads(strip_json(out)).get("pong") is True
        print(f"  live call to {live.model} ok")
    else:
        print("  offline client correct; set DEEPSEEK_API_KEY to test live")


def test_repair_loop_uses_validator_errors():
    """
    A scripted client that returns a broken case first and a fixed one
    second. Proves the repair prompt actually carries the errors and that
    the loop accepts the repair.
    """
    good = offline_case(3, seed=7)
    bad = json.loads(json.dumps(good))
    bad["culprit"] = "nobody"

    class Scripted(Deepseek):
        def __init__(self):
            super().__init__(offline=False, verbose=False)
            self.seen = []
            self.n = 0

        def chat(self, system, user, **kw):
            self.seen.append(user)
            self.n += 1
            return json.dumps(bad if self.n == 1 else good)

    client = Scripted()
    gen = CaseGenerator(client, verbose=False)
    case, report = gen.generate(n_suspects=3, seed=7)

    assert not validate(case)
    assert report["calls"] == 2 and report["repairs"] == 1, report
    assert "not one of the suspects" in client.seen[1], \
        "the repair prompt did not carry the validator's errors"
    print(f"  broken -> repaired in {report['calls']} calls, "
          f"errors were passed back")


# ======================================================================
# STAGE 11
# ======================================================================

def _played_world(bench, case, turns=3):
    w = bench.world(case)
    session = Interrogation.Session()
    ids = list(w["suspects"])
    for i in range(turns):
        s = w["suspects"][ids[i % len(ids)]]
        q = case["probe_questions"][i % len(case["probe_questions"])]
        answer, _ = Interrogation.ask_with_composure(
            s, session, case, w["tl"], w["ex"], w["ret"], q, max_tokens=25)
    return w, session


def test_transcript_and_truth_summary(bench, case):
    w, session = _played_world(bench, case, turns=3)

    tx = build_transcript(w["suspects"])
    assert "Detective:" in tx and len(tx) > 60, tx[:200]

    truth = truth_summary(case)
    assert case["culprit"] in truth
    assert "lie 0" in truth
    # the transcript must NOT leak the ground truth
    assert "SECRET:" not in tx
    print(f"  transcript {len(tx)} chars, truth summary {len(truth)} chars, "
          f"no secrets leaked into the transcript")


def test_judge_offline(bench, case):
    w, session = _played_world(bench, case, turns=3)
    judge = CaseJudge(Deepseek(offline=True, verbose=False), verbose=False)

    right = judge.judge(case, w["suspects"], case["culprit"], session)
    assert right.correct is True
    assert right.judged_by == "offline"
    assert right.culprit == case["culprit"]

    others = [s["id"] for s in case["suspects"] if s["id"] != case["culprit"]]
    wrong = judge.judge(case, w["suspects"], others[0], session)
    assert wrong.correct is False
    print(f"  {right.summary()}")


def test_judge_separates_lucky_from_earned(bench, case):
    """
    The distinction the whole stage exists for. An accusation with no
    contradictions logged against the accused is not evidence-backed.
    """
    w = bench.world(case)
    empty = Interrogation.Session()
    judge = CaseJudge(Deepseek(offline=True, verbose=False), verbose=False)

    guess = judge.judge(case, w["suspects"], case["culprit"], empty)
    assert guess.correct is True
    assert guess.evidence_backed is False, \
        "an accusation with no supporting work must not count as earned"
    print("  correct name with no work -> correct but not evidence-backed")


def test_judge_live(bench, case):
    if not ONLINE:
        print("  SKIPPED - set DEEPSEEK_API_KEY to judge with the model")
        return
    w, session = _played_world(bench, case, turns=3)
    judge = CaseJudge(verbose=False)
    v = judge.judge(case, w["suspects"], case["culprit"], session)
    assert v.judged_by.startswith("deepseek"), v.judged_by
    assert v.verdict, "the model returned an empty verdict"
    print(f"  {v.summary()}")
    print(f"  {v.verdict[:110]}")


# ======================================================================
# STAGE 12
# ======================================================================

def test_e1_measures_the_saving(bench):
    cfg = {"cases": 1, "suspects": [2, 3], "turns": 2, "repeats": 1}
    out = Experiments.e1_cache_savings(bench, cfg, resume=False)
    rows = out["rows"]
    assert len(rows) >= 2, rows

    for r in rows:
        assert r["naive_prefill"] > r["forked_prefill"], r
        assert r["saved"] > 0

    by_n = {r["n_suspects"]: r for r in rows}
    if 2 in by_n and 3 in by_n:
        assert by_n[3]["saved"] > by_n[2]["saved"], \
            "the saving must grow with suspect count"
    print("  saving is positive and grows with suspect count: "
          + ", ".join(f"n={r['n_suspects']} {r['saved_pct']:.0f}%" for r in rows))


def test_play_policies_run(bench, case):
    for policy in ("argmax", "random", "single"):
        co.COUNTER.reset()
        row = Experiments.play(bench, case, policy, max_turns=2, seed=0)
        assert row["policy"] == policy
        assert row["prefill_tokens"] > 0
        assert 0.0 <= row["recall"] <= 1.0
        assert row["accused"] in [s["id"] for s in case["suspects"]]
        print(f"  {policy:7s} found {row['contradictions_found']}"
              f"/{row['contradictions_available']} "
              f"prefill={row['prefill_tokens']}")


def test_e3_pressure_changes_the_tiers(bench, case):
    """A generous budget keeps everything on the GPU; a tiny one cannot."""
    cfg = {"cases": 1, "suspects": [3], "turns": 2, "repeats": 1}
    Experiments.CASES_DIR = HERE / "does_not_exist"
    out = Experiments.e3_eviction(bench, cfg, resume=False)
    rows = out["rows"]
    assert rows, "no eviction rows produced"

    big = max(rows, key=lambda r: r["budget_mb"])
    small = min(rows, key=lambda r: r["budget_mb"])
    assert big["tier_gpu"] >= small["tier_gpu"], (big, small)
    assert small["tier_cpu"] + small["tier_int8"] + small["tier_dropped"] > 0, \
        "a tiny budget should have pushed something off the GPU"
    print(f"  {big['budget_mb']}MB -> gpu={big['tier_gpu']} | "
          f"{small['budget_mb']}MB -> gpu={small['tier_gpu']} "
          f"int8={small['tier_int8']} cpu={small['tier_cpu']} "
          f"dropped={small['tier_dropped']}")


def test_e4_restores_the_weights(bench, case):
    """An ablation that leaves the globals mutated would poison later runs."""
    before = dict(Branching.WEIGHTS)
    cfg = {"cases": 1, "suspects": [3], "turns": 2, "repeats": 1}
    saved_grid = Experiments.WEIGHT_GRID
    Experiments.WEIGHT_GRID = saved_grid[:2]
    try:
        out = Experiments.e4_weight_ablation(bench, cfg, resume=False)
    finally:
        Experiments.WEIGHT_GRID = saved_grid

    assert out["rows"], "no ablation rows"
    assert dict(Branching.WEIGHTS) == before, \
        f"weights left mutated: {Branching.WEIGHTS} != {before}"
    print(f"  {len(out['rows'])} runs, global weights restored")


def test_plots_and_report():
    made = Experiments.make_plots()
    report = Experiments.write_report()
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Stage 12 results" in text
    print(f"  plots: {', '.join(made) or '(none)'}")
    print(f"  report: {report.name}, {len(text)} chars")


# ======================================================================

if __name__ == "__main__":
    print("=" * 62)
    print(f"STAGE 10  generator and validator   "
          f"[{'ONLINE' if ONLINE else 'offline'}]")
    print("=" * 62)
    for t in (test_validator_accepts_good_cases,
              test_validator_catches_each_kind_of_break,
              test_generator_offline_batch,
              test_deepseek_client_shape,
              test_repair_loop_uses_validator_errors):
        print(f"\n{t.__name__}")
        t()

    print("\n" + "=" * 62)
    print("STAGE 11  the judge")
    print("=" * 62)
    bench = Experiments.Bench()
    case = json.loads((HERE / "Case_Ashfield.json").read_text(encoding="utf-8"))

    for t in (test_transcript_and_truth_summary, test_judge_offline,
              test_judge_separates_lucky_from_earned, test_judge_live):
        print(f"\n{t.__name__}")
        t(bench, case)

    print("\n" + "=" * 62)
    print("STAGE 12  experiments")
    print("=" * 62)
    print("\ntest_e1_measures_the_saving")
    test_e1_measures_the_saving(bench)
    for t in (test_play_policies_run, test_e3_pressure_changes_the_tiers,
              test_e4_restores_the_weights):
        print(f"\n{t.__name__}")
        t(bench, case)
    print("\ntest_plots_and_report")
    test_plots_and_report()

    print("\n" + "=" * 62)
    print("stages 10, 11 and 12 PASSED")
    print(co.COUNTER.summary())