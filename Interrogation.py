"""
Interrogation.py   -  Stage 7
-----------------------------
Three things that make the interrogation feel like an interrogation, and
each of them leans on the cache work from earlier stages.

COMPOSURE
  One number per suspect, 1.0 calm down to 0.0 cornered. It falls when
  they are caught in a contradiction or pressed hard, and recovers a
  little when the questioning eases off. It feeds the sampling
  temperature, so a rattled suspect genuinely speaks differently rather
  than being told to act rattled.

  Rewinding restores it, because every turn records the composure that
  preceded it.

CROSS-SUSPECT QUOTING
  Telling suspect A what suspect B said puts that fact into A's context
  AND grants A the entity knowledge behind it. That changes what
  retrieval is allowed to hand them from that point on.

  The grant is recorded on the turn, so rewinding past it takes the
  knowledge away again. That is a sharp test of whether rewind is real:
  a prompt-level "forget that" could never do it.

CONTRADICTION LOG
  Every answer is checked against the suspect's earlier statements and
  against the timeline. Hits accumulate per suspect and drive composure.
"""

from dataclasses import dataclass, field

import Branching

# how far temperature can climb as composure falls
MAX_TEMPERATURE = 0.9

# composure changes
HIT_PENALTY = 0.25        # caught in a contradiction
PRESSURE_PENALTY = 0.08   # confronted with evidence
RECOVERY = 0.05           # an easy question


def temperature_for(composure):
    """
    Calm suspects answer the same way every time; cornered ones do not.

    composure 1.0 -> 0.0  (greedy, deterministic)
    composure 0.0 -> MAX_TEMPERATURE

    Speculative previews in stage 5 always pass temperature=0 regardless,
    because their scores have to be reproducible. Only committed answers
    use this.
    """
    return round(MAX_TEMPERATURE * (1.0 - max(0.0, min(1.0, composure))), 3)


@dataclass
class Hit:
    """One detected contradiction."""
    suspect_id: str
    turn_index: int
    note: str
    question: str
    answer: str


@dataclass
class Session:
    """
    Shared state across the whole interrogation: who has been caught out,
    and what each suspect has been told about the others.
    """
    hits: list = field(default_factory=list)
    quotes: list = field(default_factory=list)

    # ---------------------------------------------------------------
    def check_answer(self, suspect, case, timeline, extractor,
                     question, answer, turn_index):
        """
        Look for a contradiction and adjust composure accordingly.
        Returns the note, or None.
        """
        note = Branching.find_contradiction(
            answer, suspect, case, timeline, extractor
        )
        if note:
            self.hits.append(Hit(suspect.id, turn_index, note, question, answer))
            suspect.composure = max(0.0, suspect.composure - HIT_PENALTY)
        else:
            suspect.composure = min(1.0, suspect.composure + RECOVERY)
        return note

    def apply_pressure(self, suspect):
        """Confronting someone with evidence rattles them a little."""
        suspect.composure = max(0.0, suspect.composure - PRESSURE_PENALTY)

    def hits_for(self, suspect_id):
        return [h for h in self.hits if h.suspect_id == suspect_id]

    def most_contradicted(self, suspect_ids):
        counts = {sid: len(self.hits_for(sid)) for sid in suspect_ids}
        return max(counts, key=counts.get) if counts else None

    # ---------------------------------------------------------------
    def quote(self, source, source_turn_index, target, case, extractor):
        """
        Tell `target` what `source` said in one of their turns.

        Returns (question_text, granted_entity_ids). The caller passes the
        question to target.ask(); the grants are applied to target.knows
        and recorded on the resulting turn so a rewind can undo them.
        """
        turn = source.turns[source_turn_index]
        mentioned = extractor.extract(turn.answer)

        public = set(case["public_facts"])
        granted = sorted(e for e in mentioned
                         if e not in target.knows and e not in public)

        question = (f"{source.name} told me: \"{turn.answer}\" "
                    f"What do you say to that?")

        self.quotes.append({
            "from": source.id, "turn": source_turn_index,
            "to": target.id, "granted": granted,
        })
        return question, granted


def ask_with_composure(suspect, session, case, timeline, extractor, retriever,
                       question, granted=None, max_tokens=60, seed=0):
    """
    One full turn: retrieve, generate at the suspect's current
    temperature, check for a contradiction, and record any knowledge
    granted so that rewinding can take it away again.

    Returns (answer, note).
    """
    granted = list(granted or [])

    # apply the grant BEFORE retrieval, so newly learned facts are usable
    for eid in granted:
        if eid not in suspect.knows:
            suspect.knows.append(eid)

    allowed = suspect.knows + list(case["public_facts"])
    hits = retriever.retrieve(question, allowed, top_k=3)
    lines = retriever.as_text(hits)

    temp = temperature_for(suspect.composure)
    answer, _, _ = suspect.ask(
        question, retrieved_lines=lines,
        max_tokens=max_tokens, temperature=temp, seed=seed,
    )

    turn = suspect.turns[-1]
    turn.granted = granted

    note = session.check_answer(
        suspect, case, timeline, extractor, question, answer, turn.index
    )
    return answer, note