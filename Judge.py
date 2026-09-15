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

SYSTEM = """You are marking a detective's interrogation, and then
coaching them.

You see the true case file, the full transcript, and the accusation. The
detective never saw the true case file.

Return ONE JSON object and nothing else:

{
  "correct": true/false,
  "evidence_backed": true/false,
  "grade": "A" | "B" | "C" | "D" | "F",
  "confidence": 0.0-1.0,
  "contradictions_surfaced": ["each lie the detective actually got out,
                              in plain words"],
  "contradictions_missed": ["lies that were planted but never came out"],
  "wasted_lines": ["questions that produced nothing and why"],
  "best_moment": "the single most effective question, and what made it work",
  "what_went_wrong": ["if the accusation was wrong: the specific decisions
                      that led them astray. If it was right but thin: what
                      was missing from the case they built. Be concrete -
                      name the turn or the suspect."],
  "missed_opportunities": ["specific moments where the culprit was
                           catchable and the detective moved on. Quote the
                           answer that should have been pushed on."],
  "how_to_improve": ["3-5 concrete, actionable things to do differently.
                     Not 'ask better questions' - say WHICH question,
                     about WHAT, to WHOM, and why it would have worked."],
  "the_solution": "how the case actually breaks: which suspect, which lie,
                   and the exact line of questioning that exposes it",
  "verdict": "3-5 sentences addressed to the player"
}

evidence_backed means the accusation FOLLOWS from what the detective
established. A correct name with no supporting work is a lucky guess:
correct=true, evidence_backed=false. Be strict - that distinction is the
point of the exercise.

Grade on the WORK, not the outcome. A detective who reasoned well and
was unlucky beats one who guessed right. Roughly: A caught the culprit
in a contradiction and accused on it. C got there without really proving
it, or built a good case on the wrong person. F accused at random.

Judge only what is in the transcript - do not credit them for things the
case file says but nobody uncovered. But in "the_solution" and
"how_to_improve" you MAY use the true case file, because that is
teaching, not marking."""


# ======================================================================

@dataclass
class Verdict:
    correct: bool
    evidence_backed: bool
    confidence: float
    grade: str = "C"
    contradictions_surfaced: list = field(default_factory=list)
    contradictions_missed: list = field(default_factory=list)
    wasted_lines: list = field(default_factory=list)
    best_moment: str = ""
    what_went_wrong: list = field(default_factory=list)
    missed_opportunities: list = field(default_factory=list)
    how_to_improve: list = field(default_factory=list)
    the_solution: str = ""
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
        return (f"{mark} ({how}) grade {self.grade} - "
                f"{len(self.contradictions_surfaced)} surfaced, "
                f"{len(self.contradictions_missed)} missed [{self.judged_by}]")


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
    The debrief when there is no API key.

    Cruder than the model, but not useless: it has the ground truth, so
    it can say exactly which lie went uncaught, who was never questioned,
    and what would have exposed the culprit. That is the part a player
    actually wants after losing.
    """
    ex = FactExtractor(case["entities"])
    culprit = case["culprit"]
    by_id = {sp["id"]: sp for sp in case["suspects"]}
    live = suspects if isinstance(suspects, dict) else {}

    def said(sid):
        s = live.get(sid)
        return " ".join(t.answer for t in s.turns) if s and s.turns else ""

    def asked(sid):
        s = live.get(sid)
        return " ".join(t.question for t in s.turns) if s and s.turns else ""

    def turns_of(sid):
        s = live.get(sid)
        return len(s.turns) if s else 0

    surfaced, missed, missed_ids = [], [], []
    for sid, spec in by_id.items():
        seen = ex.extract(said(sid)) | ex.extract(asked(sid))
        for lie in spec.get("lies", []):
            label = f"{spec['name']}: claimed {lie['claim']}"
            if any(e in seen for e in lie["exposed_by"]):
                surfaced.append(label)
            else:
                missed.append(label)
                missed_ids.append((sid, lie))

    logged = len(session.hits_for(accused)) if session else 0
    backed = logged > 0
    correct = accused == culprit

    # ---- what went wrong -------------------------------------------
    wrong, missed_ops, improve = [], [], []

    if not correct:
        c = by_id[culprit]
        n = turns_of(culprit)
        if n == 0:
            wrong.append(f"You never questioned {c['name']}, who did it.")
            improve.append(f"Question every suspect at least once. "
                           f"{c['name']} was never asked anything.")
        elif n <= 2:
            wrong.append(f"You questioned {c['name']} only {n} time(s) and "
                         f"moved on. The culprit rarely breaks on the first "
                         f"answer.")
            improve.append(f"Push {c['name']} harder - press the same claim "
                           f"twice, then confront them with evidence.")
        if logged == 0:
            wrong.append(f"You accused {by_id[accused]['name']} without "
                         f"catching them in a single contradiction.")
        else:
            wrong.append(f"{by_id[accused]['name']} contradicted themselves "
                         f"{logged} time(s), but lying about where you were "
                         f"is not the same as murder - the others were "
                         f"hiding smaller things.")
            improve.append("A caught liar is not automatically the culprit. "
                           "Check what the lie was hiding before you accuse.")
    elif not backed:
        wrong.append("Right name, but the transcript does not show you "
                     "proving it - nothing was logged against them.")
        improve.append("Before accusing, get one contradiction on record "
                       "against your suspect. Use verify statement in the "
                       "lab on their last answer.")

    # ---- what was catchable and got away ---------------------------
    for sid, lie in missed_ids:
        spec = by_id[sid]
        outside = [e for e in lie["exposed_by"]
                   if e not in spec.get("knows", [])
                   and e not in case.get("public_facts", [])]
        if not outside:
            continue
        names = ", ".join(case["entities"][e]["text"] for e in outside
                          if e in case["entities"])
        missed_ops.append(
            f"{spec['name']} claimed {lie['claim']}, but in fact "
            f"{lie['truth']}. Confronting them with: {names}")

    never = [sp["name"] for sid, sp in by_id.items() if turns_of(sid) == 0]
    if never:
        improve.append(f"Never questioned: {', '.join(never)}.")
    if len(surfaced) == 0 and missed:
        improve.append("Use the lab. 'verify statement' takes a suspect's "
                       "last answer apart against the timeline and finds "
                       "contradictions you would otherwise miss.")
    improve.append("Quote one suspect at another. It rattles them and it "
                   "hands them a fact they did not have, which often pulls "
                   "out something new.")

    # ---- the solution ----------------------------------------------
    c = by_id[culprit]
    c_lie = (c.get("lies") or [{}])[0]
    solution = (f"{c['name']} did it. They claim {c_lie.get('claim', '?')}, "
                f"when really {c_lie.get('truth', '?')}. It breaks by "
                f"putting this in front of them: "
                + ", ".join(case["entities"][e]["text"]
                            for e in c_lie.get("exposed_by", [])
                            if e in case["entities"]))

    # ---- grade the work, not the outcome ---------------------------
    if correct and backed and len(surfaced) >= 2:
        grade = "A"
    elif correct and backed:
        grade = "B"
    elif correct or len(surfaced) >= 2:
        grade = "C"
    elif len(surfaced) >= 1:
        grade = "D"
    else:
        grade = "F"

    if correct and backed:
        verdict = (f"You named {c['name']} and the transcript backs it - "
                   f"{logged} contradiction(s) on record against them.")
    elif correct:
        verdict = (f"Right name, thin case. Nothing was logged against "
                   f"{c['name']}, so this reads as a guess that landed.")
    else:
        verdict = (f"You accused {by_id[accused]['name']}; it was "
                   f"{c['name']}. {len(missed)} planted lie(s) were never "
                   f"exposed.")
    verdict += " (Judged without an API key - set DEEPSEEK_API_KEY for a proper read of your questioning.)"

    return Verdict(
        correct=correct, evidence_backed=bool(backed), confidence=0.5,
        grade=grade,
        contradictions_surfaced=surfaced,
        contradictions_missed=missed,
        wasted_lines=[],
        best_moment="(not assessed without an API key)",
        what_went_wrong=wrong,
        missed_opportunities=missed_ops,
        how_to_improve=improve,
        the_solution=solution,
        verdict=verdict,
        accused=accused, culprit=culprit,
        turns=sum(turns_of(sid) for sid in by_id),
        judged_by="offline",
    )


# ======================================================================

class CaseJudge:
    def __init__(self, client=None, verbose=True):
        self.client = client or Deepseek(verbose=verbose)
        self.verbose = verbose

    def judge(self, case, suspects, accused, session=None,
              require_api=False) -> Verdict:
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
            SYSTEM, user, require=require_api,
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
            grade=str(data.get("grade", "C"))[:1].upper(),
            contradictions_surfaced=list(data.get("contradictions_surfaced", [])),
            contradictions_missed=list(data.get("contradictions_missed", [])),
            wasted_lines=list(data.get("wasted_lines", [])),
            best_moment=str(data.get("best_moment", "")),
            what_went_wrong=list(data.get("what_went_wrong", [])),
            missed_opportunities=list(data.get("missed_opportunities", [])),
            how_to_improve=list(data.get("how_to_improve", [])),
            the_solution=str(data.get("the_solution", "")),
            verdict=str(data.get("verdict", "")),
            accused=accused,
            culprit=case["culprit"],
            turns=n_turns,
            judged_by=source,
        )