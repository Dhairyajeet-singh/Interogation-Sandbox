"""
Judge.py   -  Stage 11
----------------------
Scores a finished interrogation against the truth.

WHY A MODEL AND NOT A STRING COMPARISON
---------------------------------------
"Did they name the right person" is one line of Python. It is also the
least interesting thing about a game of detective.

What is worth judging is the quality of the detective work, and in
particular the difference between:

    an accusation the player EARNED - they surfaced the contradiction,
    checked the alibi, and the accusation follows from what they
    established

    an accusation they GUESSED - right name, no supporting work

Those look identical in the final answer and completely different in the
transcript. Separating them needs a reader, which is what the model is
for. So the judge reports solve rate and evidence-backed solve rate
separately, and stage 12 plots both.

The judge sees the ground truth. The suspects never do. Keeping that
asymmetry is the whole reason the true case lives in its own reference
store that the game never reads from.

There is a mechanical fallback for when there is no API key. It is
deliberately cruder - it counts what it can count and refuses to guess at
the rest - so its output is never mistaken for the model's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from Deepseek import Deepseek
from Extractor import FactExtractor
from Retrieval import Timeline

SYSTEM = """You are marking a detective's interrogation.

You see the true case file, the full transcript, and the accusation. The
detective never saw the true case file.

Return ONE JSON object and nothing else:

{
  "correct": true/false,
  "evidence_backed": true/false,
  "confidence": 0.0-1.0,
  "contradictions_surfaced": ["short description of each lie the detective
                              actually got the suspect to reveal or that
                              was exposed during questioning"],
  "contradictions_missed": ["lies that were planted but never came out"],
  "wasted_lines": ["lines of questioning that produced nothing useful"],
  "best_moment": "the single most effective question and why",
  "verdict": "3-5 sentences addressed to the player"
}

evidence_backed means the accusation FOLLOWS from what the detective
established in the transcript. A correct name with no supporting work is
a lucky guess: correct=true, evidence_backed=false. Be strict about this
distinction - it is the point of the exercise.

Judge only what is in the transcript. Do not credit the detective for
things the true case file says but nobody uncovered."""


# ======================================================================

@dataclass
class Verdict:
    correct: bool
    evidence_backed: bool
    confidence: float
    contradictions_surfaced: list = field(default_factory=list)
    contradictions_missed: list = field(default_factory=list)
    wasted_lines: list = field(default_factory=list)
    best_moment: str = ""
    verdict: str = ""
    accused: str = ""
    culprit: str = ""
    turns: int = 0
    judged_by: str = "offline"

    def as_dict(self):
        return asdict(self)

    def summary(self):
        mark = "CORRECT" if self.correct else "WRONG"
        how = "evidence-backed" if self.evidence_backed else "unsupported"
        return (f"{mark} ({how}) - {len(self.contradictions_surfaced)} "
                f"surfaced, {len(self.contradictions_missed)} missed "
                f"[{self.judged_by}]")


# ======================================================================

def build_transcript(suspects):
    """
    Flatten the interrogation into something readable.

    `suspects` is the dict of live Suspect objects, or any object with
    .name and .turns.
    """
    lines = []
    for s in suspects.values() if isinstance(suspects, dict) else suspects:
        if not s.turns:
            continue
        lines.append(f"--- {s.name} ({s.role}) ---")
        for t in s.turns:
            lines.append(f"Detective: {t.question}")
            lines.append(f"{s.name}: {t.answer}")
            if t.granted:
                lines.append(f"[detective revealed: {', '.join(t.granted)}]")
        lines.append("")
    return "\n".join(lines).strip()


def truth_summary(case):
    """The ground truth, compact enough to sit in a prompt."""
    lines = [f"Culprit: {case['culprit']}"]
    for s in case["suspects"]:
        lines.append(f"\n{s['name']} ({s['id']}), guilty={s['guilty']}")
        lines.append(f"  really: {s['secret']}")
        for i, lie in enumerate(s.get("lies", [])):
            lines.append(f"  lie {i}: claims '{lie['claim']}' but "
                         f"'{lie['truth']}' - exposed by "
                         f"{', '.join(lie['exposed_by'])}")
    return "\n".join(lines)


# ======================================================================
# mechanical fallback
# ======================================================================

def mechanical_verdict(case, suspects, accused, session=None):
    """
    What can be worked out without a reader.

    Counts contradictions the system itself detected, and calls an
    accusation evidence-backed only if at least one contradiction was
    logged against the accused. Crude on purpose.
    """
    ex = FactExtractor(case["entities"])
    culprit = case["culprit"]

    surfaced, missed = [], []
    for spec in case["suspects"]:
        s = suspects.get(spec["id"]) if isinstance(suspects, dict) else None
        said = " ".join(t.answer for t in s.turns) if s and s.turns else ""
        mentioned = ex.extract(said) if said else set()

        for lie in spec.get("lies", []):
            # a lie counts as surfaced if the suspect was asked about it
            # and something that exposes it came up in the exchange
            asked = " ".join(t.question for t in s.turns) if s and s.turns else ""
            hit_ids = ex.extract(asked) | mentioned
            if any(e in hit_ids for e in lie["exposed_by"]):
                surfaced.append(f"{spec['name']}: {lie['claim']}")
            else:
                missed.append(f"{spec['name']}: {lie['claim']}")

    logged = len(session.hits_for(accused)) if session else 0
    backed = logged > 0 or any(accused in s for s in surfaced)

    return Verdict(
        correct=(accused == culprit),
        evidence_backed=bool(backed),
        confidence=0.5,
        contradictions_surfaced=surfaced,
        contradictions_missed=missed,
        wasted_lines=[],
        best_moment="(not assessed offline)",
        verdict=("Judged mechanically, with no API key. The accusation was "
                 f"{'right' if accused == culprit else 'wrong'} and "
                 f"{logged} contradiction(s) were logged against them."),
        accused=accused, culprit=culprit,
        turns=sum(len(s.turns) for s in suspects.values())
        if isinstance(suspects, dict) else 0,
        judged_by="offline",
    )


# ======================================================================

class CaseJudge:
    def __init__(self, client=None, verbose=True):
        self.client = client or Deepseek(verbose=verbose)
        self.verbose = verbose

    def judge(self, case, suspects, accused, session=None) -> Verdict:
        transcript = build_transcript(suspects)
        n_turns = sum(len(s.turns) for s in (
            suspects.values() if isinstance(suspects, dict) else suspects))

        if not transcript:
            v = mechanical_verdict(case, suspects, accused, session)
            v.verdict = "No questions were asked, so there is nothing to judge."
            return v

        user = (
            f"TRUE CASE FILE\n{truth_summary(case)}\n\n"
            f"TRANSCRIPT\n{transcript}\n\n"
            f"The detective accused: {accused}\n"
            f"Turns used: {n_turns}"
        )

        data, source = self.client.chat_json(
            SYSTEM, user,
            fallback=lambda: mechanical_verdict(
                case, suspects, accused, session).as_dict(),
        )

        if source == "offline":
            v = Verdict(**{k: val for k, val in data.items()
                           if k in Verdict.__annotations__})
            v.judged_by = "offline"
            return v

        return Verdict(
            correct=bool(data.get("correct")),
            evidence_backed=bool(data.get("evidence_backed")),
            confidence=float(data.get("confidence", 0.5)),
            contradictions_surfaced=list(data.get("contradictions_surfaced", [])),
            contradictions_missed=list(data.get("contradictions_missed", [])),
            wasted_lines=list(data.get("wasted_lines", [])),
            best_moment=str(data.get("best_moment", "")),
            verdict=str(data.get("verdict", "")),
            accused=accused,
            culprit=case["culprit"],
            turns=n_turns,
            judged_by=source,
        )