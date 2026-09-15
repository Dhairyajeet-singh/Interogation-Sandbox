"""
Generator.py   -  Stage 10
--------------------------
Write new cases with DeepSeek, then refuse to accept them until they
actually hold together.

THE VALIDATOR IS THE POINT
--------------------------
Asking a model for "a murder mystery with three suspects" gets you
something that reads beautifully and collapses the moment a detective
pushes on it: the culprit has no opportunity, two people are in the same
room at once and one of them is also somewhere else, a planted lie can
never be discovered. Prose quality and logical soundness are different
problems, and only the second one matters here.

So the generator is a loop:

    ask -> validate -> if it fails, hand the errors back and ask for a
    repair -> validate again -> give up after a few rounds

The repair loop is cheaper than regenerating because the model keeps the
parts that were fine. The cost is that a stubborn failure burns several
calls, so there is a hard attempt limit.

WHAT GETS CHECKED
-----------------
Structural  every required field present, ids well formed
Referential every id referenced somewhere actually exists
Temporal    times parse, ranges run forwards, nobody is in two places at
            the same moment
Solvable    the culprit had opportunity, and every planted lie is
            reachable by a detective who does not already know it
Extractable every entity has enough aliases, and no alias is ambiguous
            across two entities - that one silently breaks stage 3b

Run:
    py Generator.py --n 5 --suspects 3        write 5 cases to cases/
    py Generator.py --offline --n 3           no API, deterministic
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from Deepseek import Deepseek, DeepseekError
from Extractor import FactExtractor

HERE = Path(__file__).parent
CASES_DIR = HERE / "cases"
TEMPLATE = HERE / "Case_Ashfield.json"

TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
MAX_REPAIRS = 3


# ======================================================================
# validation
# ======================================================================

def minutes(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def validate(case) -> list[str]:
    """
    Every reason this case is not playable. Empty list means it is.

    Returns strings written for the model to act on, because they get
    handed straight back to it in the repair prompt.
    """
    errors = []
    E = lambda msg: errors.append(msg)

    # ---------------------------------------------------- structure
    for field in ("case_id", "title", "culprit", "public_brief", "entities",
                  "public_facts", "timeline", "suspects"):
        if field not in case:
            E(f"missing top-level field {field!r}")
    if errors:
        return errors

    ents = case["entities"]
    suspects = case["suspects"]
    sus_ids = [s.get("id") for s in suspects]

    if len(suspects) < 2:
        E("need at least 2 suspects")
    if len(set(sus_ids)) != len(sus_ids):
        E(f"duplicate suspect ids: {sus_ids}")

    # ---------------------------------------------------- entities
    valid_types = {"evidence", "place", "person", "time"}
    for eid, ent in ents.items():
        if not re.match(r"^(ev|loc|per|time)_[a-z0-9_]+$", eid):
            E(f"entity id {eid!r} must look like ev_x / loc_x / per_x / time_x")
        for k in ("type", "text", "aliases"):
            if k not in ent:
                E(f"entity {eid!r} missing {k!r}")
        if ent.get("type") not in valid_types:
            E(f"entity {eid!r} has type {ent.get('type')!r}, "
              f"must be one of {sorted(valid_types)}")
        aliases = ent.get("aliases") or []
        if len(aliases) < 3:
            E(f"entity {eid!r} has {len(aliases)} aliases, needs at least 3 "
              f"so the extractor can recognise it when phrased differently")

    # aliases must be unambiguous, or stage 3b breaks silently
    seen = {}
    for eid, ent in ents.items():
        for alias in ent.get("aliases") or []:
            key = alias.lower().strip()
            if key in seen and seen[key] != eid:
                E(f"alias {alias!r} is claimed by both {seen[key]} and {eid}; "
                  f"aliases must be unique across all entities")
            seen[key] = eid

    # ---------------------------------------------------- references
    for eid in case["public_facts"]:
        if eid not in ents:
            E(f"public_facts names {eid!r}, which is not an entity")

    if case["culprit"] not in sus_ids:
        E(f"culprit {case['culprit']!r} is not one of the suspects {sus_ids}")

    guilty = [s["id"] for s in suspects if s.get("guilty")]
    if guilty != [case["culprit"]]:
        E(f"exactly one suspect must have guilty=true and it must be the "
          f"culprit; got guilty={guilty}, culprit={case['culprit']!r}")

    for s in suspects:
        for k in ("id", "name", "role", "guilty", "persona", "secret",
                  "knows", "lies"):
            if k not in s:
                E(f"suspect {s.get('id')!r} missing {k!r}")
        for eid in s.get("knows", []):
            if eid not in ents:
                E(f"suspect {s.get('id')!r} knows {eid!r}, not an entity")
        if f"per_{s.get('id')}" not in ents:
            E(f"every suspect needs a person entity; add per_{s.get('id')}")

    # ---------------------------------------------------- timeline
    by_actor = {}
    for i, e in enumerate(case["timeline"]):
        for k in ("actor", "place", "from", "to"):
            if k not in e:
                E(f"timeline[{i}] missing {k!r}")
                break
        else:
            if e["actor"] not in ents:
                E(f"timeline[{i}] actor {e['actor']!r} is not an entity")
            if e["place"] not in ents:
                E(f"timeline[{i}] place {e['place']!r} is not an entity")
            for k in ("from", "to"):
                if not TIME_RE.match(e[k]):
                    E(f"timeline[{i}] {k}={e[k]!r} must be 24-hour HH:MM")
            if TIME_RE.match(e["from"]) and TIME_RE.match(e["to"]):
                if minutes(e["from"]) >= minutes(e["to"]):
                    E(f"timeline[{i}] runs backwards: "
                      f"{e['from']} to {e['to']}")
                by_actor.setdefault(e["actor"], []).append(
                    (minutes(e["from"]), minutes(e["to"]), e["place"], i))

    # nobody can be in two places at once
    for actor, spans in by_actor.items():
        spans.sort()
        for (a0, a1, pa, ia), (b0, b1, pb, ib) in zip(spans, spans[1:]):
            if b0 < a1 and pa != pb:
                E(f"{actor} is in two places at once: {pa} until "
                  f"{a1 // 60:02d}:{a1 % 60:02d} but {pb} from "
                  f"{b0 // 60:02d}:{b0 % 60:02d} "
                  f"(timeline entries {ia} and {ib})")

    # ---------------------------------------------------- solvable
    scene = case.get("crime_scene")
    window = case.get("crime_window")
    if scene and window and TIME_RE.match(window.get("from", "")) \
            and TIME_RE.match(window.get("to", "")):
        if scene not in ents:
            E(f"crime_scene {scene!r} is not an entity")
        w0, w1 = minutes(window["from"]), minutes(window["to"])
        actor = f"per_{case['culprit']}"
        spans = by_actor.get(actor, [])
        had = any(p == scene and s < w1 and e > w0 for s, e, p, _ in spans)
        if not had:
            E(f"the culprit {actor} is never at the crime scene {scene} "
              f"during {window['from']}-{window['to']}, so they had no "
              f"opportunity; add a timeline entry placing them there")

    # every planted lie must be discoverable
    for s in suspects:
        lies = s.get("lies") or []
        if not lies:
            E(f"suspect {s.get('id')!r} has no lies, so there is nothing "
              f"for the detective to catch them on")
        for j, lie in enumerate(lies):
            for k in ("claim", "truth", "exposed_by"):
                if k not in lie:
                    E(f"suspect {s.get('id')!r} lie[{j}] missing {k!r}")
            exposed = lie.get("exposed_by") or []
            unknown = [e for e in exposed if e not in ents]
            if unknown:
                E(f"suspect {s.get('id')!r} lie[{j}] exposed_by names "
                  f"{unknown}, which are not entities")
            public = set(case.get("public_facts") or [])
            outside = [e for e in exposed
                       if e in ents and e not in s.get("knows", [])
                       and e not in public]
            if exposed and not outside:
                E(f"suspect {s.get('id')!r} lie[{j}] can only be exposed by "
                  f"facts they already know, so confronting them with it is "
                  f"no surprise; add evidence outside their knows list")

    # the extractor must be able to build from this
    try:
        FactExtractor(ents)
    except Exception as exc:
        E(f"the fact extractor rejects these entities: {exc}")

    return errors


# ======================================================================
# prompts
# ======================================================================

SYSTEM = """You design murder-mystery cases for an interrogation game.

You return ONE JSON object and nothing else. No prose, no markdown fences.

The case must be logically airtight. A detective will interrogate the
suspects and try to catch them lying, so every lie you plant must be
discoverable from evidence the liar does not already have, and the
culprit must have had the opportunity to commit the crime.

Schema:

{
  "case_id": "snake_case_id",
  "title": "short title",
  "culprit": "<suspect id>",
  "crime_scene": "<loc_ entity id>",
  "crime_window": {"from": "HH:MM", "to": "HH:MM"},
  "public_brief": "What everyone knows. 2-4 sentences, then: 'You are being questioned by a detective. Answer only from what you personally know. Never invent facts. Keep every answer to one or two short sentences, in character.'",
  "entities": {
    "<id>": {"type": "evidence|place|person|time", "text": "one sentence",
             "aliases": ["at least three different ways someone might say this"]}
  },
  "public_facts": ["entity ids everyone knows"],
  "timeline": [
    {"actor": "per_x", "place": "loc_y", "from": "HH:MM", "to": "HH:MM",
     "verifiable_by": "<entity id or null>"}
  ],
  "suspects": [
    {"id": "x", "name": "Full Name", "role": "job", "guilty": false,
     "persona": "second person, their manner under questioning",
     "secret": "SECRET: what they are really hiding and what they will claim instead",
     "knows": ["entity ids only this suspect knows"],
     "lies": [{"claim": "what they say", "truth": "what happened",
               "exposed_by": ["entity ids that catch them"]}]}
  ],
  "probe_questions": ["5 opening questions a detective might ask"]
}

Hard rules:
- entity ids start with ev_, loc_, per_ or time_
- every suspect needs a matching per_<id> person entity
- every entity needs at least 3 aliases, and no alias may be used by two entities
- exactly one suspect has guilty=true, and it is the culprit
- the culprit must appear in the timeline at crime_scene during crime_window
- no actor may be in two different places at overlapping times
- every suspect has at least one lie
- each lie's exposed_by must include at least one entity NOT in that
  suspect's knows list
- time entities need an "HH:MM" string among their aliases"""


def generate_prompt(n_suspects, seed_hint=""):
    return (f"Write a new case with exactly {n_suspects} suspects. "
            f"Make the setting, names and crime different from anything "
            f"obvious. {seed_hint}".strip())


def repair_prompt(case, errors):
    listed = "\n".join(f"- {e}" for e in errors)
    return ("This case failed validation. Fix ONLY these problems and "
            "return the complete corrected JSON object. Keep everything "
            "else exactly as it is.\n\n"
            f"Problems:\n{listed}\n\n"
            f"Current case:\n{json.dumps(case, indent=1)}")


# ======================================================================
# offline generator
# ======================================================================

def offline_case(n_suspects=3, seed=0):
    """
    A deterministic, always-valid case builder.

    Not a substitute for the model - the cases are formulaic. It exists so
    that stage 12's experiments and every test can run with no API key and
    no network, and so a dead API cannot take down a demo.
    """
    rng = random.Random(seed)
    people = [
        ("ambrose", "Nathaniel Ambrose", "estate steward"),
        ("keller", "Dr Ruth Keller", "physician"),
        ("voss", "Peter Voss", "chauffeur"),
        ("iyer", "Meera Iyer", "archivist"),
        ("blaine", "Sofia Blaine", "restorer"),
        ("dunn", "Harold Dunn", "gardener"),
        ("nakata", "Yuki Nakata", "curator"),
        ("ferrow", "Alice Ferrow", "secretary"),
    ]
    chosen = people[:n_suspects]
    culprit = rng.randrange(n_suspects)

    places = [("loc_library", "library", ["library", "the reading room", "among the books"]),
              ("loc_cellar", "cellar", ["cellar", "the basement", "below stairs"]),
              ("loc_terrace", "terrace", ["terrace", "the veranda", "outside at the back"]),
              ("loc_kitchen", "kitchen", ["kitchen", "the scullery", "by the stove"]),
              ("loc_gallery", "gallery", ["gallery", "the long room", "where the paintings hang"]),
              ("loc_stables", "stables", ["stables", "the horse yard", "out by the horses"]),
              ("loc_study2", "study", ["the study", "the writing room", "his desk"]),
              ("loc_attic", "attic", ["attic", "the loft", "up under the roof"]),
              ("loc_orchard", "orchard", ["orchard", "the fruit trees", "past the wall"]),
              ("loc_chapel", "chapel", ["chapel", "the old chapel", "by the altar"])]

    ents = {
        "per_victim": {"type": "person",
                       "text": "Lord Edmund Carew, the victim.",
                       "aliases": ["Edmund Carew", "Lord Carew", "the victim",
                                   "his lordship"]},
        "ev_letter": {"type": "evidence",
                      "text": "A torn letter was found under the body.",
                      "aliases": ["the letter", "torn letter", "the note",
                                  "correspondence"]},
        "ev_ledger": {"type": "evidence",
                      "text": "The estate ledger shows payments that were never declared.",
                      "aliases": ["the ledger", "account book", "the estate accounts",
                                  "the books"]},
        "ev_keyring": {"type": "evidence",
                       "text": "A keyring is missing from the steward's office.",
                       "aliases": ["the keyring", "missing keys", "set of keys",
                                   "the keys"]},
        "loc_library": {"type": "place", "text": "The library, where he was found.",
                        "aliases": places[0][2]},
    }
    for pid, _, al in places[1:n_suspects + 1]:
        ents[pid] = {"type": "place",
                     "text": f"The {al[0]}.", "aliases": al}

    times = [("time_2000", "20:00", ["20:00", "8pm", "eight", "eight o'clock"]),
             ("time_2015", "20:15", ["20:15", "8:15", "quarter past eight"]),
             ("time_2030", "20:30", ["20:30", "8:30", "half past eight"]),
             ("time_2045", "20:45", ["20:45", "8:45", "quarter to nine"]),
             ("time_2100b", "21:00", ["21:00", "9pm", "nine", "nine o'clock"])]
    for tid, clock, al in times:
        ents[tid] = {"type": "time", "text": f"{clock}, during the evening.",
                     "aliases": al}

    timeline = [{"actor": "per_victim", "place": "loc_library",
                 "from": "20:00", "to": "21:00", "verifiable_by": None}]
    suspects = []

    for i, (sid, name, role) in enumerate(chosen):
        pid = f"per_{sid}"
        ents[pid] = {"type": "person", "text": f"{name}, the {role}.",
                     "aliases": [name, name.split()[-1], f"the {role}"]}
        secret_ev = f"ev_secret_{sid}"
        ents[secret_ev] = {
            "type": "evidence",
            "text": f"{name} had a private arrangement with the victim.",
            "aliases": [f"{sid} arrangement", f"the {role} arrangement",
                        f"{name}'s dealings"]}

        guilty = (i == culprit)
        place = "loc_library" if guilty else places[i + 1][0]
        start = "20:15" if guilty else "20:00"
        end = "20:45" if guilty else "21:00"
        timeline.append({"actor": pid, "place": place, "from": start,
                         "to": end, "verifiable_by": None})
        if guilty:
            timeline.append({"actor": pid, "place": places[i + 1][0],
                             "from": "20:45", "to": "21:00",
                             "verifiable_by": None})

        suspects.append({
            "id": sid, "name": name, "role": role, "guilty": guilty,
            "persona": f"You are {name}, the {role}. You are guarded and "
                       f"answer briefly.",
            "secret": (f"SECRET: you were in the {places[i+1][1]} and will say so. "
                       if not guilty else
                       f"SECRET: you were in the library with him and struck him. "
                       f"You will claim you were in the {places[i+1][1]} the whole time. "),
            "knows": [secret_ev, place, pid],
            "lies": [{
                "claim": f"was in the {places[i+1][1]} all evening",
                "truth": ("was in the library between 20:15 and 20:45"
                          if guilty else "left briefly and told no one"),
                "exposed_by": ["ev_ledger", "ev_keyring"],
            }],
        })

    return {
        "case_id": f"offline_{n_suspects}_{seed}",
        "title": f"The Carew Case ({n_suspects} suspects)",
        "culprit": chosen[culprit][0],
        "crime_scene": "loc_library",
        "crime_window": {"from": "20:00", "to": "21:00"},
        "generated_by": "offline",
        "public_brief": (
            "Case file. Lord Edmund Carew was found dead in the library "
            "between 20:00 and 21:00 on the evening of 3 March. He was "
            "struck once from behind. You are being questioned by a "
            "detective. Answer only from what you personally know. Never "
            "invent facts. Keep every answer to one or two short "
            "sentences, in character."),
        "entities": ents,
        "public_facts": ["per_victim", "loc_library", "ev_letter",
                         "time_2000", "time_2100b"],
        "timeline": timeline,
        "suspects": suspects,
        "probe_questions": [
            "Where were you between eight and nine?",
            "When did you last see him alive?",
            "Did you go into the library at any point?",
            "What do you know about the ledger?",
            "Who do you think did this?",
        ],
    }


# ======================================================================
# the generator
# ======================================================================

class CaseGenerator:
    def __init__(self, client=None, verbose=True):
        self.client = client or Deepseek(verbose=verbose)
        self.verbose = verbose
        self.attempts = 0
        self.repairs = 0
        self.rejected = 0

    def generate(self, n_suspects=3, seed=0, hint=""):
        """
        Produce one validated case.

        Returns (case, report) where report records how many calls it took
        and what went wrong on the way - the failure rate is itself a
        result worth putting in the write-up.
        """
        report = {"n_suspects": n_suspects, "seed": seed,
                  "calls": 0, "repairs": 0, "errors_seen": [],
                  "source": None}

        if self.client.offline:
            case = offline_case(n_suspects, seed)
            errs = validate(case)
            if errs:
                raise RuntimeError(f"offline builder produced an invalid case: {errs}")
            case["generated_by"] = "offline"
            report["source"] = "offline"
            return case, report

        case = None
        for round_no in range(MAX_REPAIRS + 1):
            self.attempts += 1
            report["calls"] += 1
            try:
                if case is None:
                    raw = self.client.chat(SYSTEM, generate_prompt(n_suspects, hint))
                else:
                    self.repairs += 1
                    report["repairs"] += 1
                    raw = self.client.chat(SYSTEM, repair_prompt(case, errors))
                from Deepseek import strip_json
                case = json.loads(strip_json(raw))
            except Exception as exc:
                if self.verbose:
                    print(f"  round {round_no}: call failed - {exc}")
                report["errors_seen"].append(f"call failed: {exc}")
                continue

            errors = validate(case)
            if not errors:
                case["generated_by"] = self.client.model
                case.setdefault("case_id", f"gen_{n_suspects}_{seed}")
                report["source"] = self.client.model
                if self.verbose:
                    print(f"  valid after {report['calls']} call(s)")
                return case, report

            report["errors_seen"].extend(errors)
            if self.verbose:
                print(f"  round {round_no}: {len(errors)} problem(s), repairing")
                for e in errors[:4]:
                    print(f"      {e}")

        self.rejected += 1
        raise RuntimeError(
            f"could not produce a valid case in {MAX_REPAIRS + 1} rounds; "
            f"last errors: {errors[:5]}")

    def batch(self, n=5, suspects=(3,), out_dir=CASES_DIR):
        """Write n validated cases to disk and report the failure rate."""
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        written, reports = [], []

        for i in range(n):
            k = suspects[i % len(suspects)]
            if self.verbose:
                print(f"\ncase {i + 1}/{n} ({k} suspects)")
            try:
                case, report = self.generate(n_suspects=k, seed=i)
            except RuntimeError as exc:
                if self.verbose:
                    print(f"  giving up: {exc}")
                reports.append({"seed": i, "n_suspects": k, "failed": True})
                continue

            path = out_dir / f"{case['case_id']}.json"
            path.write_text(json.dumps(case, indent=2), encoding="utf-8")
            written.append(path)
            reports.append(report)

        total_calls = sum(r.get("calls", 0) for r in reports)
        repairs = sum(r.get("repairs", 0) for r in reports)
        first_try = sum(1 for r in reports
                        if not r.get("failed") and r.get("repairs", 0) == 0)

        summary = {
            "written": len(written),
            "requested": n,
            "total_calls": total_calls,
            "repair_calls": repairs,
            "valid_first_try": first_try,
            "first_try_rate": first_try / n if n else 0.0,
            "reports": reports,
        }
        return written, summary


# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--suspects", type=int, nargs="+", default=[3])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=str(CASES_DIR))
    args = ap.parse_args()

    gen = CaseGenerator(Deepseek(offline=args.offline or None))
    written, summary = gen.batch(args.n, tuple(args.suspects), args.out)

    print("\n" + "=" * 58)
    print(f"wrote {summary['written']}/{summary['requested']} cases to {args.out}")
    print(f"calls {summary['total_calls']} "
          f"({summary['repair_calls']} were repairs)")
    print(f"valid first try: {summary['valid_first_try']}/{summary['requested']} "
          f"({summary['first_try_rate']:.0%})")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()