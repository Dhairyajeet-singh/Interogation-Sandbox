"""
Branching.py   -  Stage 5
-------------------------
Three-option questioning.

Before the detective commits to a question, the suspect's cache is forked
three times and a different candidate question is tried in each fork. The
three trial answers are scored, all three are shown to the player, and
whichever is picked becomes the suspect's real state. The other two forks
are discarded.

This is only affordable because forking is nearly free. Three trial
questions the naive way would mean three full re-reads of the case file.

SCORING IS MECHANICAL. No model judges anything. Three counted quantities:

  coverage     entity ids in the answer that have not come up before
  verifiable   claims the timeline graph could adjudicate (time/place/person)
  contradiction  1 if the answer conflicts with an earlier statement or
                 the timeline, else 0

Weights are tunable and get grid-searched against ground truth in stage 12,
so they are defended by measurement rather than by argument.
"""

from dataclasses import dataclass, field

import cache_ops as co

# tuned in stage 12; these are the starting point
WEIGHTS = {"coverage": 0.45, "verifiable": 0.25, "contradiction": 0.30}

# caps used to normalise raw counts into 0..1
CAPS = {"coverage": 4.0, "verifiable": 3.0}

PREVIEW_TOKENS = 36     # trial answers are short; the full answer comes later


# ======================================================================
# candidate generation
# ======================================================================

def generate_candidates(suspect, case, timeline, extractor, n=3):
    """
    Build candidate questions from unresolved case state, not from the
    model. Asking a 0.5B model to "write three probing questions" returns
    three bland rephrasings; templates filled from the graph do not.

    Four template types, in priority order:
      1. press an unverified claim the suspect already made
      2. confront with evidence that exposes one of their lies
      3. revisit a gap between two of their own statements
      4. open a topic they have not been asked about yet
    """
    already_asked = {t.question for t in suspect.turns}
    mentioned = set()
    for t in suspect.turns:
        mentioned |= extractor.extract(t.answer)

    candidates = []

    # --- 1. press a claim they made that the timeline can check
    for t in reversed(suspect.turns):
        claims = extractor.verifiable_claims(t.answer)
        places = [c for c in claims if case["entities"][c]["type"] == "place"]
        times = [c for c in claims if case["entities"][c]["type"] == "time"]
        if places and times:
            p = case["entities"][places[0]]["aliases"][0]
            tm = case["entities"][times[0]]["aliases"][0]
            q = f"You said you were at the {p} around {tm}. Who saw you there?"
            if q not in already_asked:
                candidates.append(("press_claim", q))
                break

    # --- 2. confront with evidence that exposes a lie
    for lie in suspect.lies:
        for eid in lie["exposed_by"]:
            if eid in suspect.knows or eid in case["public_facts"]:
                continue                     # they already know it; no shock
            ent = case["entities"][eid]
            q = f"We have {ent['aliases'][0]}. {ent['text']} Explain that."
            if q not in already_asked:
                candidates.append(("confront_evidence", q))
                break
        if len(candidates) >= 2:
            break

    # --- 3. revisit a gap between two of their own statements
    if len(suspect.turns) >= 2:
        q = ("Earlier you told me something different. "
             "Walk me through that evening again, in order.")
        if q not in already_asked:
            candidates.append(("revisit_gap", q))

    # --- 4. open an untouched topic
    for eid in suspect.knows:
        if eid in mentioned:
            continue
        ent = case["entities"][eid]
        q = f"Tell me about {ent['aliases'][0]}."
        if q not in already_asked:
            candidates.append(("open_topic", q))
        if len(candidates) >= n + 2:
            break

    # --- fallback so we always have n options
    generic = [
        "Where exactly were you when she died?",
        "Who else was in the house that night?",
        "Is there anything you have not told me?",
    ]
    for g in generic:
        if len(candidates) >= n:
            break
        if g not in already_asked:
            candidates.append(("generic", g))

    return candidates[:n]


# ======================================================================
# scoring
# ======================================================================

def find_contradiction(answer, suspect, case, timeline, extractor):
    """
    Does this answer conflict with something already said, or with the
    timeline? Returns an explanation string, or None.
    """
    claims = extractor.extract(answer)
    places = [c for c in claims if case["entities"][c]["type"] == "place"]
    times = [c for c in claims if case["entities"][c]["type"] == "time"]

    actor = "per_" + suspect.id

    # against the timeline graph
    for p in places:
        for tm in times:
            clock = _clock_of(case["entities"][tm])
            if clock is None:
                continue
            verdict, why = timeline.check_alibi(actor, p, clock)
            if verdict == "contradicted":
                return f"timeline: {why}"

    # against their own earlier statements
    for turn in suspect.turns:
        earlier = extractor.extract(turn.answer)
        earlier_places = {c for c in earlier
                          if case["entities"][c]["type"] == "place"}
        now_places = set(places)
        if earlier_places and now_places and not (earlier_places & now_places):
            a = sorted(earlier_places)[0]
            b = sorted(now_places)[0]
            return f"said {a} earlier, now says {b}"

    return None


def _clock_of(entity):
    """Pull an HH:MM string out of a time entity's aliases, if there is one."""
    for alias in entity["aliases"]:
        if ":" in alias and alias.replace(":", "").isdigit():
            h, m = alias.split(":")
            if len(h) <= 2 and len(m) == 2:
                return f"{int(h):02d}:{m}"
    return None


@dataclass
class Candidate:
    kind: str
    question: str
    answer: str
    cache: object = None
    cached_ids: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)
    coverage: int = 0
    verifiable: int = 0
    contradiction: int = 0
    contradiction_note: str = ""
    score: float = 0.0
    new_entities: set = field(default_factory=set)

    def band(self):
        """Coarse label for the UI. The raw score goes in the cache log."""
        if self.score >= 0.60:
            return "high"
        if self.score >= 0.30:
            return "medium"
        return "low"


def score_candidate(cand, suspect, case, timeline, extractor):
    """Fill in the counted quantities and the combined score."""
    found = extractor.extract(cand.answer)

    seen = set()
    for t in suspect.turns:
        seen |= extractor.extract(t.answer)

    cand.new_entities = found - seen
    cand.coverage = len(cand.new_entities)
    cand.verifiable = len(extractor.verifiable_claims(cand.answer))

    note = find_contradiction(cand.answer, suspect, case, timeline, extractor)
    cand.contradiction = 1 if note else 0
    cand.contradiction_note = note or ""

    cov = min(cand.coverage / CAPS["coverage"], 1.0)
    ver = min(cand.verifiable / CAPS["verifiable"], 1.0)

    cand.score = (WEIGHTS["coverage"] * cov
                  + WEIGHTS["verifiable"] * ver
                  + WEIGHTS["contradiction"] * cand.contradiction)
    return cand


# ======================================================================
# the branch step
# ======================================================================

def explore(suspect, case, timeline, extractor, retriever, n=3,
            preview_tokens=PREVIEW_TOKENS):
    """
    Fork the suspect n times, try a different question in each, score the
    answers, and return the candidates sorted best first.

    Nothing is committed. The caller picks one and passes it to commit().

    Greedy decoding (temperature=0) so the scores are reproducible: run
    this twice on the same state and you get the same numbers.
    """
    specs = generate_candidates(suspect, case, timeline, extractor, n=n)
    before_forks = co.COUNTER.count("fork")
    before_prefill = co.COUNTER.prefill_tokens

    out = []
    for kind, question in specs:
        allowed = suspect.knows + list(case["public_facts"])
        hits = retriever.retrieve(question, allowed, top_k=3)
        lines = retriever.as_text(hits)

        trial = co.fork(suspect.cache, label=f"{suspect.id}:try:{kind}")
        answer, new_cache, new_ids = suspect.ask(
            question, retrieved_lines=lines,
            max_tokens=preview_tokens, temperature=0.0,
            cache=trial, cached_ids=suspect.cached_ids, record=False,
        )
        cand = Candidate(kind=kind, question=question, answer=answer,
                         cache=new_cache, cached_ids=new_ids,
                         retrieved=lines)
        out.append(score_candidate(cand, suspect, case, timeline, extractor))

    out.sort(key=lambda c: c.score, reverse=True)

    cost = co.COUNTER.prefill_tokens - before_prefill
    forks = co.COUNTER.count("fork") - before_forks
    return out, {"forks": forks, "prefill_tokens": cost}


def commit(suspect, candidate, retriever, case, max_tokens=60):
    """
    Accept a candidate as the suspect's real next turn.

    The trial answer was generated from a fork of the live cache, so the
    fork already holds exactly the right state. We adopt it rather than
    regenerating, which means the answer the player saw is the answer
    they get - the score was not a promise about some other text.
    """
    snap_key = f"{suspect.id}:t{len(suspect.turns)}"
    suspect.store.put(snap_key, suspect.cache)

    from Suspects import Turn
    suspect.turns.append(Turn(
        index=len(suspect.turns),
        question=candidate.question,
        answer=candidate.answer,
        ids_before=suspect.cached_ids,
        retrieved=candidate.retrieved,
        snapshot_key=snap_key,
        composure_before=suspect.composure,
    ))
    suspect.cache = candidate.cache
    suspect.cached_ids = list(candidate.cached_ids)
    return candidate.answer