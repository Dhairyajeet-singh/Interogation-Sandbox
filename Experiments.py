"""
Experiments.py   -  Stage 12
----------------------------
The measurements. This is the stage that makes the project a project
rather than a demo, so it is the one not to let get squeezed.

Four experiments, each answering a question somebody will actually ask:

E1  "How much does the cache actually save?"
    Prefill tokens with forking vs without, swept over suspect count.
    Forking stays flat; re-reading climbs linearly. The headline plot.

E2  "Does trying three questions beat asking one?"
    Contradictions surfaced within a fixed turn budget, comparing
    argmax over three scored forks, random choice among the three, and
    single-shot. If argmax does not beat random, the scoring function
    does not work - and that is a finding too.

E3  "What does the memory budget cost you?"
    Snapshot hit rate and tier distribution as the GPU budget shrinks,
    plus how often a rewind has to fall back to recomputation.

E4  "Where did those scoring weights come from?"
    Grid search over the coverage / verifiable / contradiction weights
    against ground truth. The answer to the question an examiner will
    ask, in the form of a table rather than an argument.

Everything is checkpointed to results/ after each unit of work, so a long
run can be interrupted and resumed. On a 4 GB card the large scale takes
a few hours; start it and leave it.

    py Experiments.py --scale small     ~10 min, for checking it runs
    py Experiments.py --scale large     the real thing
    py Experiments.py --only e1 e4      just some of them
    py Experiments.py --plots           redraw from saved results
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

import Branching
import cache_ops as co
import Interrogation
from Extractor import FactExtractor
from Generator import offline_case, validate
from Model_setup import load_model
from Retrieval import HybridRetriever, Timeline
from Snapshots import SnapshotStore, GPU, INT8, CPU, DROPPED
from Suspects import (Suspect, compute_shared_len, make_shared_block,
                      make_private_block)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CASES_DIR = HERE / "cases"

SCALES = {
    "small":  {"cases": 3,  "suspects": [2, 3, 4],
               "turns": 4, "repeats": 1},
    "medium": {"cases": 5,  "suspects": [2, 3, 4, 5, 6],
               "turns": 6, "repeats": 2},
    "large":  {"cases": 10, "suspects": [2, 3, 4, 5, 6, 7, 8],
               "turns": 8, "repeats": 3},
}


# ======================================================================
# harness
# ======================================================================

class Bench:
    """Loads the model once and builds a fresh world per case."""

    def __init__(self, verbose=True):
        self.tok, self.model, self.device = load_model()
        self.verbose = verbose

    def world(self, case, budget_bytes=None):
        """Prefill the shared block and fork one suspect per spec."""
        ex = FactExtractor(case["entities"])
        tl = Timeline(case["timeline"])
        ret = HybridRetriever(case["entities"], tl)
        store = SnapshotStore(gpu_budget_bytes=budget_bytes,
                              device=self.device)

        shared_block = make_shared_block(case)
        privates = [make_private_block(case, sp) for sp in case["suspects"]]
        n_shared, shared_ids = compute_shared_len(self.tok, shared_block,
                                                  privates)
        shared, _ = co.prefill(
            self.model, torch.tensor([shared_ids], device=self.device),
            label="shared")

        suspects = {
            sp["id"]: Suspect(sp, case, shared, shared_ids, self.model,
                              self.tok, self.device, store=store)
            for sp in case["suspects"]
        }
        return {"case": case, "ex": ex, "tl": tl, "ret": ret,
                "store": store, "suspects": suspects,
                "n_shared": n_shared, "shared_ids": shared_ids}


def load_cases(scale_cfg, min_suspects=None):
    """
    Prefer generated cases on disk; fall back to the deterministic offline
    builder so the experiments always have something to run on.

    min_suspects filters out cases too small for the sweep being run, and
    tops the list up with offline cases of the right size. Without this a
    2-suspect case on disk silently drops half of E1's data points.
    """
    cases = []

    # The hand-written case goes first. Its lies are constructed so a
    # detective can actually catch them; the offline builder's are
    # formulaic, which matters a lot for E2 and E4.
    hand = HERE / "Case_Ashfield.json"
    if hand.exists():
        c = json.loads(hand.read_text(encoding="utf-8"))
        if not validate(c) and not (min_suspects
                                    and len(c["suspects"]) < min_suspects):
            cases.append(c)

    if CASES_DIR.exists():
        for p in sorted(CASES_DIR.glob("*.json")):
            c = json.loads(p.read_text(encoding="utf-8"))
            if validate(c):
                continue
            if min_suspects and len(c["suspects"]) < min_suspects:
                continue
            cases.append(c)

    size = min_suspects or 3
    i = 0
    while len(cases) < scale_cfg["cases"]:
        cases.append(offline_case(size, seed=100 + i))
        i += 1
    return cases[:scale_cfg["cases"]]


def checkpoint(name, data):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=2),
                                          encoding="utf-8")


def load_checkpoint(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ======================================================================
# E1 - what the cache saves
# ======================================================================

def e1_cache_savings(bench, cfg, resume=True):
    """
    Prefill tokens with forking vs the naive baseline, by suspect count.

    naive = every suspect reads the shared block itself.
    forked = it is read once and copied.

    Deliberately measured in tokens, not milliseconds: on a Pascal card
    fp16 buys memory rather than speed, so wall clock is noisy while the
    token count is exact and reproducible on any machine.
    """
    saved = load_checkpoint("e1") if resume else None
    rows = saved["rows"] if saved else []
    done = {(r["case_id"], r["n_suspects"]) for r in rows}

    for case in load_cases(cfg, min_suspects=max(cfg["suspects"])):
        for n in cfg["suspects"]:
            key = (case["case_id"], n)
            if key in done:
                continue

            trimmed = json.loads(json.dumps(case))
            trimmed["suspects"] = trimmed["suspects"][:n]
            trimmed["culprit"] = trimmed["suspects"][0]["id"]
            for i, sp in enumerate(trimmed["suspects"]):
                sp["guilty"] = (i == 0)

            co.COUNTER.reset()
            w = bench.world(trimmed)

            forked = co.COUNTER.prefill_tokens
            per_private = [s.n_private for s in w["suspects"].values()]
            naive = sum(w["n_shared"] + p for p in per_private)

            rows.append({
                "case_id": case["case_id"],
                "n_suspects": n,
                "shared_tokens": w["n_shared"],
                "forked_prefill": forked,
                "naive_prefill": naive,
                "saved": naive - forked,
                "saved_pct": 100.0 * (naive - forked) / naive if naive else 0,
                "bytes_per_token": co.bytes_per_token(
                    list(w["suspects"].values())[0].cache),
            })
            checkpoint("e1", {"rows": rows})
            if bench.verbose:
                r = rows[-1]
                print(f"  {case['case_id'][:22]:22s} n={n} "
                      f"forked={r['forked_prefill']:6d} "
                      f"naive={r['naive_prefill']:6d} "
                      f"saved={r['saved_pct']:.0f}%")

    return {"rows": rows}


# ======================================================================
# E2 - does branching beat asking one question
# ======================================================================

def play(bench, case, policy, max_turns, seed=0, budget=None):
    """
    Auto-play one interrogation under a policy and report what it found.

    policies
      argmax  explore three forks, take the highest-scoring
      random  explore three forks, take one at random
      single  take the first candidate without exploring the others
    """
    rng = random.Random(seed)
    w = bench.world(case, budget_bytes=budget)
    session = Interrogation.Session()
    ids = list(w["suspects"])

    prefill_before = co.COUNTER.prefill_tokens

    for turn in range(max_turns):
        s = w["suspects"][ids[turn % len(ids)]]

        if policy == "single":
            specs = Branching.generate_candidates(
                s, case, w["tl"], w["ex"], n=1)
            if not specs:
                continue
            _, question = specs[0]
            allowed = s.knows + list(case["public_facts"])
            lines = w["ret"].as_text(
                w["ret"].retrieve(question, allowed, top_k=3))
            answer, _, _ = s.ask(question, retrieved_lines=lines,
                                 max_tokens=40, temperature=0.0)
            session.check_answer(s, case, w["tl"], w["ex"],
                                 question, answer, s.turns[-1].index)
            continue

        cands, _ = Branching.explore(s, case, w["tl"], w["ex"], w["ret"], n=3)
        if not cands:
            continue
        pick = 0 if policy == "argmax" else rng.randrange(len(cands))
        chosen = cands[pick]
        answer = Branching.commit(s, chosen, w["ret"], case)
        session.check_answer(s, case, w["tl"], w["ex"],
                             chosen.question, answer, s.turns[-1].index)

    available = sum(len(sp.get("lies", [])) for sp in case["suspects"])
    accused = session.most_contradicted(ids) or ids[0]

    return {
        "policy": policy,
        "case_id": case["case_id"],
        "seed": seed,
        "turns": max_turns,
        "contradictions_found": len(session.hits),
        "contradictions_available": available,
        "recall": len(session.hits) / available if available else 0.0,
        "accused": accused,
        "culprit": case["culprit"],
        "correct": accused == case["culprit"],
        "prefill_tokens": co.COUNTER.prefill_tokens - prefill_before,
        "store": w["store"].report(),
    }


def e2_branching_value(bench, cfg, resume=True):
    saved = load_checkpoint("e2") if resume else None
    rows = saved["rows"] if saved else []
    done = {(r["case_id"], r["policy"], r["seed"]) for r in rows}

    for case in load_cases(cfg):
        for policy in ("argmax", "random", "single"):
            for seed in range(cfg["repeats"]):
                if (case["case_id"], policy, seed) in done:
                    continue
                co.COUNTER.reset()
                row = play(bench, case, policy, cfg["turns"], seed=seed)
                rows.append(row)
                checkpoint("e2", {"rows": rows})
                if bench.verbose:
                    print(f"  {case['case_id'][:20]:20s} {policy:7s} "
                          f"seed={seed} found={row['contradictions_found']}"
                          f"/{row['contradictions_available']} "
                          f"{'correct' if row['correct'] else 'wrong'}")

    return {"rows": rows}


# ======================================================================
# E3 - what the memory budget costs
# ======================================================================

def e3_eviction(bench, cfg, resume=True):
    """
    Shrink the GPU snapshot budget and watch what happens to the hit rate
    and the tier mix. The last column is the one that matters: how often a
    rewind found nothing and had to rebuild.
    """
    saved = load_checkpoint("e3") if resume else None
    rows = saved["rows"] if saved else []
    done = {(r["case_id"], r["budget_mb"]) for r in rows}

    cases = load_cases(cfg)[:max(1, cfg["cases"] // 2)]
    budgets_mb = [512, 128, 64, 32, 16, 8, 4, 2, 1]

    for case in cases:
        for mb in budgets_mb:
            if (case["case_id"], mb) in done:
                continue
            co.COUNTER.reset()
            w = bench.world(case, budget_bytes=mb * 1024 * 1024)
            ids = list(w["suspects"])

            # fill the store, then rewind into it
            for turn in range(cfg["turns"]):
                s = w["suspects"][ids[turn % len(ids)]]
                s.ask(case["probe_questions"][turn % len(case["probe_questions"])],
                      max_tokens=25)

            # Capture the tier mix BEFORE rewinding. A rewind consumes
            # its snapshot and the store forgets it, so measuring after
            # would show an empty store no matter how much pressure there
            # was.
            st = w["store"]
            counts = st.tier_counts()

            recomputed = 0
            for s in w["suspects"].values():
                if not s.turns:
                    continue
                tier = s.restore_snapshot(0)
                if tier == "recomputed":
                    recomputed += 1
            rows.append({
                "case_id": case["case_id"],
                "budget_mb": mb,
                "puts": st.stats.puts,
                "gets": st.stats.gets,
                "hit_rate": st.stats.hit_rate,
                "hits_gpu": st.stats.hits_gpu,
                "hits_int8": st.stats.hits_int8,
                "hits_cpu": st.stats.hits_cpu,
                "misses": st.stats.misses,
                "tier_gpu": counts[GPU], "tier_int8": counts[INT8],
                "tier_cpu": counts[CPU], "tier_dropped": counts[DROPPED],
                "demotions": dict(st.stats.demotions),
                "recomputes": recomputed,
                "evict_events": co.COUNTER.count("evict"),
            })
            checkpoint("e3", {"rows": rows})
            if bench.verbose:
                r = rows[-1]
                print(f"  {case['case_id'][:20]:20s} budget={mb:4d}MB "
                      f"hit={r['hit_rate']:.0%} "
                      f"gpu/int8/cpu={r['hits_gpu']}/{r['hits_int8']}/{r['hits_cpu']} "
                      f"recompute={r['recomputes']}")

    return {"rows": rows}


# ======================================================================
# E4 - where the scoring weights come from
# ======================================================================

WEIGHT_GRID = [
    {"coverage": 1.00, "verifiable": 0.00, "contradiction": 0.00},
    {"coverage": 0.00, "verifiable": 1.00, "contradiction": 0.00},
    {"coverage": 0.00, "verifiable": 0.00, "contradiction": 1.00},
    {"coverage": 0.45, "verifiable": 0.25, "contradiction": 0.30},
    {"coverage": 0.34, "verifiable": 0.33, "contradiction": 0.33},
    {"coverage": 0.20, "verifiable": 0.20, "contradiction": 0.60},
    {"coverage": 0.60, "verifiable": 0.20, "contradiction": 0.20},
]


def e4_weight_ablation(bench, cfg, resume=True):
    """
    Try each weighting, auto-play with argmax, and see which finds the
    most planted contradictions.

    This is what you say when someone asks where 0.45 came from: it won a
    grid search against ground truth, and here is the table.
    """
    saved = load_checkpoint("e4") if resume else None
    rows = saved["rows"] if saved else []
    done = {(r["case_id"], json.dumps(r["weights"], sort_keys=True), r["seed"])
            for r in rows}

    original = dict(Branching.WEIGHTS)
    cases = load_cases(cfg)[:max(1, cfg["cases"] // 2)]

    try:
        for weights in WEIGHT_GRID:
            key_w = json.dumps(weights, sort_keys=True)
            for case in cases:
                for seed in range(cfg["repeats"]):
                    if (case["case_id"], key_w, seed) in done:
                        continue
                    Branching.WEIGHTS.clear()
                    Branching.WEIGHTS.update(weights)
                    co.COUNTER.reset()
                    row = play(bench, case, "argmax", cfg["turns"], seed=seed)
                    row["weights"] = weights
                    rows.append(row)
                    checkpoint("e4", {"rows": rows})
                    if bench.verbose:
                        print(f"  {key_w[:44]:44s} {case['case_id'][:14]:14s} "
                              f"found={row['contradictions_found']}")
    finally:
        Branching.WEIGHTS.clear()
        Branching.WEIGHTS.update(original)

    return {"rows": rows}


# ======================================================================
# plots
# ======================================================================

def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS.mkdir(exist_ok=True)
    made = []

    # ---------------------------------------------------------- E1
    e1 = load_checkpoint("e1")
    if e1 and e1["rows"]:
        by_n = {}
        for r in e1["rows"]:
            by_n.setdefault(r["n_suspects"], []).append(r)
        ns = sorted(by_n)
        forked = [sum(x["forked_prefill"] for x in by_n[n]) / len(by_n[n])
                  for n in ns]
        naive = [sum(x["naive_prefill"] for x in by_n[n]) / len(by_n[n])
                 for n in ns]

        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(ns, naive, "o-", label="naive: each suspect re-reads the case")
        ax.plot(ns, forked, "s-", label="forked: read once, copied")
        ax.set_xlabel("suspects")
        ax.set_ylabel("prefill tokens")
        ax.set_title("E1  prefill cost vs suspect count")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "e1_cache_savings.png", dpi=150)
        plt.close(fig)
        made.append("e1_cache_savings.png")

    # ---------------------------------------------------------- E2
    e2 = load_checkpoint("e2")
    if e2 and e2["rows"]:
        by_p = {}
        for r in e2["rows"]:
            by_p.setdefault(r["policy"], []).append(r)
        order = [p for p in ("argmax", "random", "single") if p in by_p]
        recall = [sum(x["recall"] for x in by_p[p]) / len(by_p[p]) for p in order]
        solve = [sum(1 for x in by_p[p] if x["correct"]) / len(by_p[p])
                 for p in order]

        fig, ax = plt.subplots(figsize=(7, 4.4))
        x = range(len(order))
        ax.bar([i - 0.18 for i in x], recall, width=0.36,
               label="contradiction recall")
        ax.bar([i + 0.18 for i in x], solve, width=0.36, label="solve rate")
        ax.set_xticks(list(x))
        ax.set_xticklabels(order)
        ax.set_ylim(0, 1)
        ax.set_title("E2  does choosing among three forks help?")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(RESULTS / "e2_branching_value.png", dpi=150)
        plt.close(fig)
        made.append("e2_branching_value.png")

    # ---------------------------------------------------------- E3
    e3 = load_checkpoint("e3")
    if e3 and e3["rows"]:
        by_b = {}
        for r in e3["rows"]:
            by_b.setdefault(r["budget_mb"], []).append(r)
        budgets = sorted(by_b, reverse=True)
        hit = [sum(x["hit_rate"] for x in by_b[b]) / len(by_b[b]) for b in budgets]
        gpu = [sum(x["tier_gpu"] for x in by_b[b]) / len(by_b[b]) for b in budgets]
        i8 = [sum(x["tier_int8"] for x in by_b[b]) / len(by_b[b]) for b in budgets]
        cpu = [sum(x["tier_cpu"] for x in by_b[b]) / len(by_b[b]) for b in budgets]

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        a1.plot([str(b) for b in budgets], hit, "o-")
        a1.set_xlabel("GPU budget (MB)")
        a1.set_ylabel("snapshot hit rate")
        a1.set_ylim(0, 1.05)
        a1.set_title("E3a  hit rate under pressure")
        a1.grid(alpha=0.3)

        labels = [str(b) for b in budgets]
        a2.bar(labels, gpu, label="gpu")
        a2.bar(labels, i8, bottom=gpu, label="int8")
        a2.bar(labels, cpu, bottom=[g + i for g, i in zip(gpu, i8)], label="cpu")
        a2.set_xlabel("GPU budget (MB)")
        a2.set_ylabel("snapshots per tier")
        a2.set_title("E3b  where snapshots end up")
        a2.legend()
        a2.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(RESULTS / "e3_eviction.png", dpi=150)
        plt.close(fig)
        made.append("e3_eviction.png")

    # ---------------------------------------------------------- E4
    e4 = load_checkpoint("e4")
    if e4 and e4["rows"]:
        by_w = {}
        for r in e4["rows"]:
            by_w.setdefault(json.dumps(r["weights"], sort_keys=True), []).append(r)
        keys = sorted(by_w, key=lambda k: -sum(
            x["recall"] for x in by_w[k]) / len(by_w[k]))
        recall = [sum(x["recall"] for x in by_w[k]) / len(by_w[k]) for k in keys]
        labels = [json.loads(k) for k in keys]
        short = [f"{w['coverage']:.2f}/{w['verifiable']:.2f}/"
                 f"{w['contradiction']:.2f}" for w in labels]

        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.barh(range(len(short)), recall)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("contradiction recall")
        ax.set_title("E4  weights (coverage / verifiable / contradiction)")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(RESULTS / "e4_weight_ablation.png", dpi=150)
        plt.close(fig)
        made.append("e4_weight_ablation.png")

    return made


# ======================================================================
# report
# ======================================================================

def write_report():
    """A markdown summary of whatever results exist."""
    lines = ["# Stage 12 results", ""]

    e1 = load_checkpoint("e1")
    if e1 and e1["rows"]:
        by_n = {}
        for r in e1["rows"]:
            by_n.setdefault(r["n_suspects"], []).append(r)
        lines += ["## E1 — what the cache saves", "",
                  "| suspects | forked | naive | saved | % |",
                  "|---|---|---|---|---|"]
        for n in sorted(by_n):
            rs = by_n[n]
            f = sum(x["forked_prefill"] for x in rs) / len(rs)
            v = sum(x["naive_prefill"] for x in rs) / len(rs)
            lines.append(f"| {n} | {f:.0f} | {v:.0f} | {v-f:.0f} | "
                         f"{100*(v-f)/v:.0f}% |")
        lines.append("")

    e2 = load_checkpoint("e2")
    if e2 and e2["rows"]:
        by_p = {}
        for r in e2["rows"]:
            by_p.setdefault(r["policy"], []).append(r)
        lines += ["## E2 — does branching help", "",
                  "| policy | recall | solve rate | runs |",
                  "|---|---|---|---|"]
        for p in ("argmax", "random", "single"):
            if p not in by_p:
                continue
            rs = by_p[p]
            rec = sum(x["recall"] for x in rs) / len(rs)
            sol = sum(1 for x in rs if x["correct"]) / len(rs)
            lines.append(f"| {p} | {rec:.2f} | {sol:.2f} | {len(rs)} |")
        lines.append("")
        total_found = sum(x["contradictions_found"] for x in e2["rows"])
        if total_found == 0:
            lines += [
                "**This experiment is under-powered and answers nothing.** "
                "No contradictions fired under any policy, so the three are "
                "indistinguishable. Contradiction detection needs a suspect "
                "to name a specific place AND a specific time that conflict "
                "with the timeline; a 0.5B model gives short, vague answers "
                "instead. Re-run on the cloud with a 7-8B model before "
                "drawing any conclusion about the scoring function.",
                "",
            ]

        if "argmax" in by_p and "random" in by_p and total_found > 0:
            a = sum(x["recall"] for x in by_p["argmax"]) / len(by_p["argmax"])
            r = sum(x["recall"] for x in by_p["random"]) / len(by_p["random"])
            lines.append(
                f"argmax {'beats' if a > r else 'does not beat'} random "
                f"({a:.2f} vs {r:.2f}). "
                + ("The scoring function carries information."
                   if a > r else
                   "The scoring function is not earning its place — which "
                   "is a finding worth reporting rather than hiding.")
            )
            lines.append("")

    e3 = load_checkpoint("e3")
    if e3 and e3["rows"]:
        by_b = {}
        for r in e3["rows"]:
            by_b.setdefault(r["budget_mb"], []).append(r)
        lines += ["## E3 — memory budget", "",
                  "| budget MB | hit rate | gpu | int8 | cpu | recomputes |",
                  "|---|---|---|---|---|---|"]
        for b in sorted(by_b, reverse=True):
            rs = by_b[b]
            avg = lambda k: sum(x[k] for x in rs) / len(rs)
            lines.append(f"| {b} | {avg('hit_rate'):.0%} | "
                         f"{avg('hits_gpu'):.1f} | {avg('hits_int8'):.1f} | "
                         f"{avg('hits_cpu'):.1f} | {avg('recomputes'):.1f} |")
        lines.append("")

    e4 = load_checkpoint("e4")
    if e4 and e4["rows"]:
        by_w = {}
        for r in e4["rows"]:
            by_w.setdefault(json.dumps(r["weights"], sort_keys=True), []).append(r)
        ranked = sorted(by_w, key=lambda k: -sum(
            x["recall"] for x in by_w[k]) / len(by_w[k]))
        best_recall = sum(x["recall"] for x in by_w[ranked[0]]) / len(by_w[ranked[0]])
        lines += ["## E4 — scoring weights", ""]
        if best_recall == 0:
            lines += ["**Every weighting scored zero, so this table ranks "
                      "nothing.** Same cause as E2: the contradiction "
                      "detector rarely fires at 0.5B. The ablation is only "
                      "meaningful once the cloud run produces answers "
                      "specific enough to adjudicate.", ""]
        lines += [
                  "| coverage | verifiable | contradiction | recall |",
                  "|---|---|---|---|"]
        for k in ranked:
            w = json.loads(k)
            rec = sum(x["recall"] for x in by_w[k]) / len(by_w[k])
            lines.append(f"| {w['coverage']:.2f} | {w['verifiable']:.2f} | "
                         f"{w['contradiction']:.2f} | {rec:.2f} |")
        best = json.loads(ranked[0])
        lines += ["", f"Best: {best}. These are the defaults, and this table "
                      "is why — not an argument."]

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="small")
    ap.add_argument("--only", nargs="+", choices=["e1", "e2", "e3", "e4"])
    ap.add_argument("--plots", action="store_true",
                    help="redraw plots from saved results and exit")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore checkpoints and start over")
    args = ap.parse_args()

    if args.plots:
        made = make_plots()
        print("plots:", ", ".join(made) or "(no results yet)")
        print("report:", write_report())
        return

    cfg = SCALES[args.scale]
    which = args.only or ["e1", "e2", "e3", "e4"]
    bench = Bench()
    started = time.time()

    print(f"\nscale={args.scale} cases={cfg['cases']} "
          f"suspects={cfg['suspects']} turns={cfg['turns']} "
          f"repeats={cfg['repeats']}")

    if "e1" in which:
        print("\nE1  prefill cost vs suspect count")
        e1_cache_savings(bench, cfg, resume=not args.fresh)
    if "e2" in which:
        print("\nE2  branching vs random vs single-shot")
        e2_branching_value(bench, cfg, resume=not args.fresh)
    if "e3" in which:
        print("\nE3  eviction under a shrinking budget")
        e3_eviction(bench, cfg, resume=not args.fresh)
    if "e4" in which:
        print("\nE4  scoring weight ablation")
        e4_weight_ablation(bench, cfg, resume=not args.fresh)

    made = make_plots()
    report = write_report()
    mins = (time.time() - started) / 60
    print(f"\ndone in {mins:.1f} min")
    print("plots:", ", ".join(made) or "(none)")
    print("report:", report)


if __name__ == "__main__":
    main()