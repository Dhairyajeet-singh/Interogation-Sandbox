"""
game.py   -  Stage 4
--------------------
The interrogation loop, as a LangGraph state machine.

The graph is genuinely cyclic: after each answer it routes back to pick
another action, and that is why LangGraph earns its place here rather
than a plain while-loop. Nodes:

    choose_suspect -> retrieve -> branch -> generate -> record -> decide
                                                                    |
                             +--------------------------------------+
                             |
                        (loop back, or accuse -> END)

State carries the case, the suspects, the live cache plan, and the
transcript. Every node reads and returns a plain dict, which keeps the
graph easy to follow.

Run:  py game.py
"""

import json
import random
from typing import TypedDict, Optional, List, Dict, Any

import torch
from langgraph.graph import StateGraph, END
from Model_setup import load_model, MODEL_NAME, DEVICE

import Branching
import cache_ops as co
from Extractor import FactExtractor
from Retrieval import HybridRetriever, Timeline, KnowledgeChecker
from Suspects import (Suspect, compute_shared_len, make_shared_block,
                      make_private_block)


# ======================================================================
# state
# ======================================================================

class GameState(TypedDict):
    case: Dict[str, Any]
    suspect_id: Optional[str]
    question: Optional[str]
    retrieved: List[str]
    candidates: List[Any]
    answer: Optional[str]
    violations: List[str]
    turn_count: int
    max_turns: int
    accusation: Optional[str]
    finished: bool
    log: List[str]


# ======================================================================
# the world, built once and shared by every node
# ======================================================================

class World:
    """Everything the nodes need. Built once at startup."""

    def __init__(self, case_path="case_ashfield.json", auto=True):
        self.case = json.load(open(case_path, encoding="utf-8"))
        self.auto = auto

        self.tok, self.model, _ = load_model()

        self.extractor = FactExtractor(self.case["entities"])
        self.timeline = Timeline(self.case["timeline"])
        self.retriever = HybridRetriever(self.case["entities"], self.timeline)
        self.checker = KnowledgeChecker(self.extractor, self.case["public_facts"])

        # ---- prefill the shared block ONCE, fork it per suspect ----
        shared_block = make_shared_block(self.case)
        privates = [make_private_block(self.case, sp)
                    for sp in self.case["suspects"]]
        n_shared, shared_ids = compute_shared_len(self.tok, shared_block, privates)
        self.n_shared = n_shared
        self.shared_ids = shared_ids
        self.shared_cache, _ = co.prefill(
            self.model, torch.tensor([shared_ids], device=DEVICE),
            label="shared",
        )
        print(f"shared block prefilled once: {n_shared} tokens "
              f"({co.nbytes(self.shared_cache)/1024/1024:.2f} MB)")

        self.suspects = {
            spec["id"]: Suspect(spec, self.case, self.shared_cache,
                                shared_ids, self.model, self.tok, DEVICE)
            for spec in self.case["suspects"]
        }
        for s in self.suspects.values():
            print(f"  forked {s.name:<20} private {s.n_private:>3} tok, "
                  f"base {s.base_len}")

    def suspect(self, sid):
        return self.suspects[sid]


WORLD: Optional[World] = None


# ======================================================================
# nodes
# ======================================================================

def node_choose_suspect(state: GameState) -> GameState:
    """Pick who to question next. Round-robin in auto mode."""
    ids = list(WORLD.suspects.keys())
    if WORLD.auto:
        state["suspect_id"] = ids[state["turn_count"] % len(ids)]
    else:
        print("\nsuspects: " + ", ".join(
            f"{i}) {WORLD.suspects[x].name}" for i, x in enumerate(ids)))
        raw = input("question which suspect? [0] ").strip() or "0"
        state["suspect_id"] = ids[int(raw)]

    s = WORLD.suspect(state["suspect_id"])
    state["log"].append(f"turn {state['turn_count']}: questioning {s.name}")
    return state


def node_branch(state: GameState) -> GameState:
    """
    Stage 5. Fork the suspect three times, try three questions, score them.
    The forks cost almost no prefill because the case file is already
    in the cache they were forked from.
    """
    s = WORLD.suspect(state["suspect_id"])
    cands, cost = Branching.explore(
        s, WORLD.case, WORLD.timeline, WORLD.extractor, WORLD.retriever, n=3
    )
    state["candidates"] = cands
    state["log"].append(
        f"  explored {cost['forks']} forks for {cost['prefill_tokens']} prefill tokens"
    )

    print(f"\n--- {s.name} ---")
    for i, c in enumerate(cands):
        print(f"  [{i}] ({c.band():<6} {c.score:.2f}) {c.question}")
        print(f"        preview: {c.answer[:70]}")
        if c.contradiction:
            print(f"        ! {c.contradiction_note}")
    return state


def node_pick(state: GameState) -> GameState:
    """Choose one candidate. Auto mode takes the highest score."""
    cands = state["candidates"]
    if WORLD.auto:
        choice = 0
    else:
        raw = input(f"pick [0-{len(cands)-1}] or 'own': ").strip()
        if raw == "own":
            state["question"] = input("your question: ").strip()
            state["candidates"] = []
            return state
        choice = int(raw or 0)
    state["question"] = cands[choice].question
    state["candidates"] = [cands[choice]]
    return state


def node_generate(state: GameState) -> GameState:
    """
    Commit the chosen branch. If the player typed their own question we
    have no fork for it, so we ask normally.
    """
    s = WORLD.suspect(state["suspect_id"])

    if state["candidates"]:
        cand = state["candidates"][0]
        answer = Branching.commit(s, cand, WORLD.retriever, WORLD.case)
        state["retrieved"] = [c for c in cand.new_entities]
    else:
        allowed = s.knows + list(WORLD.case["public_facts"])
        hits = WORLD.retriever.retrieve(state["question"], allowed, top_k=3)
        lines = WORLD.retriever.as_text(hits)
        answer, _, _ = s.ask(state["question"], retrieved_lines=lines,
                             max_tokens=60, temperature=0.0)
        state["retrieved"] = hits

    state["answer"] = answer
    print(f"\n{s.name}: {answer}")
    return state


def node_record(state: GameState) -> GameState:
    """
    Check the answer did not leak knowledge the suspect should not have,
    and note any contradiction against the timeline.
    """
    s = WORLD.suspect(state["suspect_id"])
    v = WORLD.checker.violations(state["answer"], s.knows)
    state["violations"] = sorted(v)
    if v:
        print(f"  [leak] {s.name} mentioned facts outside their knowledge: "
              f"{sorted(v)}")
        state["log"].append(f"  LEAK {s.id}: {sorted(v)}")

    note = Branching.find_contradiction(
        state["answer"], s, WORLD.case, WORLD.timeline, WORLD.extractor
    )
    if note:
        print(f"  [contradiction] {note}")
        state["log"].append(f"  CONTRADICTION {s.id}: {note}")

    state["turn_count"] += 1
    return state


def node_accuse(state: GameState) -> GameState:
    """End the game with an accusation."""
    ids = list(WORLD.suspects.keys())
    if WORLD.auto:
        # auto-play: accuse whoever produced the most contradictions
        counts = {i: 0 for i in ids}
        for line in state["log"]:
            for i in ids:
                if line.startswith(f"  CONTRADICTION {i}"):
                    counts[i] += 1
        state["accusation"] = max(counts, key=counts.get)
    else:
        print("\n" + ", ".join(f"{i}) {WORLD.suspects[x].name}"
                               for i, x in enumerate(ids)))
        state["accusation"] = ids[int(input("accuse whom? ").strip() or 0)]

    truth = WORLD.case["culprit"]
    acc = state["accusation"]
    print(f"\n{'='*66}")
    print(f"you accused : {WORLD.suspects[acc].name}")
    print(f"the culprit : {WORLD.suspects[truth].name}")
    print("VERDICT     : " + ("CORRECT" if acc == truth else "WRONG"))
    state["finished"] = True
    return state


def route_after_record(state: GameState) -> str:
    """The cycle. Keep questioning until the turn budget runs out."""
    if state["turn_count"] >= state["max_turns"]:
        return "accuse"
    return "choose_suspect"


# ======================================================================
# graph
# ======================================================================

def build_graph():
    g = StateGraph(GameState)

    g.add_node("choose_suspect", node_choose_suspect)
    g.add_node("branch", node_branch)
    g.add_node("pick", node_pick)
    g.add_node("generate", node_generate)
    g.add_node("record", node_record)
    g.add_node("accuse", node_accuse)

    g.set_entry_point("choose_suspect")
    g.add_edge("choose_suspect", "branch")
    g.add_edge("branch", "pick")
    g.add_edge("pick", "generate")
    g.add_edge("generate", "record")
    g.add_conditional_edges(
        "record", route_after_record,
        {"choose_suspect": "choose_suspect", "accuse": "accuse"},
    )
    g.add_edge("accuse", END)

    return g.compile()


# ======================================================================
# main
# ======================================================================

def main(auto=True, max_turns=6):
    global WORLD
    random.seed(0)
    torch.manual_seed(0)

    WORLD = World(auto=auto)
    graph = build_graph()

    state: GameState = {
        "case": WORLD.case,
        "suspect_id": None,
        "question": None,
        "retrieved": [],
        "candidates": [],
        "answer": None,
        "violations": [],
        "turn_count": 0,
        "max_turns": max_turns,
        "accusation": None,
        "finished": False,
        "log": [],
    }

    final = graph.invoke(state, {"recursion_limit": 200})

    # ---- accounting ----
    real = co.COUNTER.prefill_tokens
    naive = 0
    for s in WORLD.suspects.values():
        naive += WORLD.n_shared + s.n_private
        for t in s.turns:
            naive += 40                       # rough per-question cost
    naive += WORLD.n_shared * (len(WORLD.suspects) - 1)

    print(f"\n{'='*66}")
    print("CACHE ACCOUNTING")
    print(f"  shared block prefilled once : {WORLD.n_shared} tokens")
    print(f"  suspects forked from it     : {len(WORLD.suspects)}")
    print(f"  saved by not re-reading it  : "
          f"{WORLD.n_shared * (len(WORLD.suspects)-1)} tokens")
    print(f"  {co.COUNTER.summary()}")
    print("\nlast cache events:")
    print(co.COUNTER.log(kinds=("fork", "crop", "snapshot"), last=12))


if __name__ == "__main__":
    import sys
    interactive = "--play" in sys.argv
    main(auto=not interactive, max_turns=6)