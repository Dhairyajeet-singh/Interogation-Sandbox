"""
Api.py   -  Stage 9, game edition
---------------------------------
FastAPI backend. All logic lives here; the React app only renders.

ONE PROCESS, ONE WORKER, NO RELOAD
    py Api.py                       correct
    uvicorn Api:app --reload        WRONG - wipes the model on every save
    uvicorn Api:app --workers 4     WRONG - each worker has its own caches

WHAT THE GAME LAYER ADDS
    turn budget      you cannot question forever; run out and you must accuse
    score            contradictions and discoveries earn, wasted turns cost
    difficulty       easy / normal / hard change the budget, hints, rewind cost
    evidence board   facts the detective has actually uncovered, grows in play
    dossiers         per-suspect pins: what they said, what caught them
    hidden previews  explore() returns question types only - the answer and
                     its score are revealed on commit, server-side, so the
                     UI cannot peek

WHAT DEEPSEEK DOES HERE
    New Case   Generator.CaseGenerator, deepseek-reasoner, with the
               validator + repair loop. Slow (60-180s), so it runs as a
               background job the UI polls.
    Accuse     Judge.CaseJudge, deepseek-reasoner, returns the earned-vs-
               lucky verdict. Also a background job.
    Both fall back to the offline path with no key, and stamp the result
    so it can never be mistaken for model output.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import Branching
import cache_ops as co
import Interrogation
from Deepseek import Deepseek, have_key
from Extractor import FactExtractor
from Generator import CaseGenerator, validate
from Judge import CaseJudge
from Model_setup import load_model
from Retrieval import HybridRetriever, Timeline, KnowledgeChecker
from Snapshots import SnapshotStore
from Suspects import (Suspect, compute_shared_len, make_shared_block,
                      make_private_block)

HERE = Path(__file__).parent
CASES_DIR = HERE / "cases"
HAND_CASE = HERE / "Case_Ashfield.json"
FRONTEND_DIST = HERE / "frontend" / "dist"
MANUAL = HERE / "MANUAL.md"

SNAPSHOT_BUDGET_MB = 96

DIFFICULTY = {
    "easy":   {"turns": 15, "hints": True,  "rewind_cost": 0, "label": "Easy"},
    "normal": {"turns": 10, "hints": True,  "rewind_cost": 1, "label": "Normal"},
    "hard":   {"turns": 7,  "hints": False, "rewind_cost": 2, "label": "Hard"},
}

SCORE = {
    "contradiction":    100,   # a suspect caught out
    "discovery":         25,   # a new fact on the evidence board
    "turn":             -10,   # every question costs
    "wasted":           -15,   # extra, for a turn that revealed nothing
    "correct_backed":   500,
    "correct_lucky":    150,
    "wrong":           -200,
}


# ======================================================================
# case library
# ======================================================================

def library():
    """Every playable case on disk, hand-written one first."""
    out = []
    paths = [HAND_CASE] if HAND_CASE.exists() else []
    if CASES_DIR.exists():
        paths += sorted(CASES_DIR.glob("*.json"))
    for p in paths:
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if validate(c):
            continue
        out.append({
            "case_id": c["case_id"],
            "title": c["title"],
            "n_suspects": len(c["suspects"]),
            "generated_by": c.get("generated_by", "unknown"),
            "path": str(p),
        })
    return out


def load_case_by_id(case_id):
    for entry in library():
        if entry["case_id"] == case_id:
            return json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
    raise HTTPException(404, f"no case {case_id!r} in the library")


# ======================================================================
# background jobs - generation and judging are slow
# ======================================================================

class Jobs:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def start(self, kind, fn):
        job_id = uuid.uuid4().hex[:10]
        with self._lock:
            self._jobs[job_id] = {"id": job_id, "kind": kind,
                                  "status": "running", "started": time.time(),
                                  "result": None, "error": None, "log": []}

        def run():
            try:
                res = fn(lambda msg: self._log(job_id, msg))
                with self._lock:
                    self._jobs[job_id].update(status="done", result=res)
            except Exception as exc:
                with self._lock:
                    self._jobs[job_id].update(status="failed", error=str(exc))

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def _log(self, job_id, msg):
        with self._lock:
            self._jobs[job_id]["log"].append(msg)

    def get(self, job_id):
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                raise HTTPException(404, f"no job {job_id}")
            return dict(j, elapsed=round(time.time() - j["started"], 1))


JOBS = Jobs()


# ======================================================================
# the world
# ======================================================================

class World:
    def __init__(self):
        self.tok, self.model, self.device = load_model()
        self.case = None
        self.difficulty = "normal"
        self.load(json.loads(HAND_CASE.read_text(encoding="utf-8")), "normal")

    # ---------------------------------------------------- setup
    def load(self, case, difficulty="normal"):
        """Build a fresh world for a case. The model stays loaded."""
        if difficulty not in DIFFICULTY:
            raise HTTPException(400, f"difficulty must be one of {list(DIFFICULTY)}")
        errs = validate(case)
        if errs:
            raise HTTPException(400, f"case is not playable: {errs[:3]}")

        self.case = case
        self.difficulty = difficulty
        self.rules = DIFFICULTY[difficulty]

        self.extractor = FactExtractor(case["entities"])
        self.timeline = Timeline(case["timeline"])
        self.retriever = HybridRetriever(case["entities"], self.timeline)
        self.checker = KnowledgeChecker(self.extractor, case["public_facts"])
        self.store = SnapshotStore(
            gpu_budget_bytes=SNAPSHOT_BUDGET_MB * 1024 * 1024,
            device=self.device)

        shared_block = make_shared_block(case)
        privates = [make_private_block(case, sp) for sp in case["suspects"]]
        self.n_shared, self.shared_ids = compute_shared_len(
            self.tok, shared_block, privates)
        co.COUNTER.reset()
        self.shared_cache, _ = co.prefill(
            self.model, torch.tensor([self.shared_ids], device=self.device),
            label="shared")

        self.suspects = {
            sp["id"]: Suspect(sp, case, self.shared_cache, self.shared_ids,
                              self.model, self.tok, self.device,
                              store=self.store)
            for sp in case["suspects"]
        }

        self.session = Interrogation.Session()
        self.candidates = {}
        self.turns_used = 0
        self.score = 0
        self.finished = False
        self.accusation = None
        self.verdict = None
        self.judge_job = None

        # the detective's own knowledge, separate from any suspect's
        self.discovered = set(case["public_facts"])
        self.board = []          # [{kind, text, entity, suspect, turn}]
        self.pins = {sid: [] for sid in self.suspects}
        self.forensics = None

    def suspect(self, sid):
        if sid not in self.suspects:
            raise HTTPException(404, f"no suspect {sid!r}")
        return self.suspects[sid]

    def tools(self):
        if self.forensics is None:
            from Mcp_client import ForensicsClient, Forensics
            self.forensics = Forensics(ForensicsClient())
        return self.forensics

    # ---------------------------------------------------- game rules
    def require_open(self):
        if self.finished:
            raise HTTPException(400, "the case is closed - accusation made")
        if self.turns_used >= self.rules["turns"]:
            raise HTTPException(400, "turn budget exhausted - you must accuse")

    def spend_turn(self, n=1):
        self.turns_used += n
        self.score += SCORE["turn"] * n

    def discover(self, entity_ids, kind, text, suspect_id=None, turn=None):
        """Put facts on the evidence board. Returns how many were new."""
        new = 0
        for eid in entity_ids:
            if eid in self.discovered or eid not in self.case["entities"]:
                continue
            self.discovered.add(eid)
            new += 1
            self.score += SCORE["discovery"]
            self.board.append({
                "kind": kind, "entity": eid,
                "text": self.case["entities"][eid]["text"],
                "suspect": suspect_id, "turn": turn, "via": text,
            })
        return new

    def pin(self, sid, kind, text, turn=None):
        self.pins[sid].append({"kind": kind, "text": text, "turn": turn})

    def record_answer(self, s, question, answer, kind=None):
        """Everything that happens after a suspect speaks."""
        turn = s.turns[-1].index if s.turns else None
        note = self.session.check_answer(
            s, self.case, self.timeline, self.extractor, question, answer, turn)

        mentioned = self.extractor.extract(answer)
        new = self.discover(mentioned, "statement",
                            f"{s.name} mentioned it", s.id, turn)

        if note:
            self.score += SCORE["contradiction"]
            self.pin(s.id, "contradiction", note, turn)
        if kind == "confront_evidence":
            self.session.apply_pressure(s)
        if new == 0 and not note:
            self.score += SCORE["wasted"]

        violations = sorted(self.checker.violations(answer, s.knows))
        return {"contradiction": note, "new_facts": new,
                "violations": violations}


WORLD: World | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global WORLD
    WORLD = World()
    print(f"ready: {WORLD.case['title']}, {len(WORLD.suspects)} suspects, "
          f"{WORLD.n_shared} shared tokens on {WORLD.device}, "
          f"deepseek={'live' if have_key() else 'OFFLINE'}")
    yield


app = FastAPI(title="Branching Interrogation Sandbox", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ======================================================================
# serialisation
# ======================================================================

def suspect_json(s):
    ents = WORLD.case["entities"]
    return {
        "id": s.id, "name": s.name, "role": s.role,
        "composure": round(s.composure, 3),
        "temperature": Interrogation.temperature_for(s.composure),
        "cache_tokens": co.length(s.cache), "base_tokens": s.base_len,
        "turns": [{
            "index": t.index, "question": t.question, "answer": t.answer,
            "cache_len_before": t.cache_len_before,
            "granted": [{"id": g, "text": ents[g]["text"]}
                        for g in t.granted if g in ents],
            "snapshot_tier": (WORLD.store.entries[t.snapshot_key].tier
                              if t.snapshot_key in WORLD.store.entries
                              else "dropped"),
            "score": getattr(t, "ui_score", None),
            "band": getattr(t, "ui_band", None),
            "kind": getattr(t, "ui_kind", None),
        } for t in s.turns],
        "hits": [{"turn": h.turn_index, "note": h.note}
                 for h in WORLD.session.hits_for(s.id)],
        "pins": WORLD.pins.get(s.id, []),
    }


def candidate_public(c, index, hints):
    """
    What the player is allowed to see BEFORE committing. No preview, no
    score. The type is a hint, and hard mode hides even that.
    """
    return {"index": index,
            "kind": c.kind if hints else "question",
            "question": c.question}


def accounting():
    n = len(WORLD.suspects)
    return {
        "prefill_tokens": co.COUNTER.prefill_tokens,
        "decode_tokens": co.COUNTER.decode_tokens,
        "forks": co.COUNTER.count("fork"), "crops": co.COUNTER.count("crop"),
        "evictions": co.COUNTER.count("evict"),
        "shared_tokens": WORLD.n_shared, "suspects": n,
        "tokens_saved_on_shared_block": WORLD.n_shared * (n - 1),
        "bytes_per_token": round(co.bytes_per_token(WORLD.shared_cache)),
    }


def store_json():
    st = WORLD.store
    return {"report": st.report(), "tiers": st.tier_counts(),
            "gpu_bytes": st.gpu_bytes(), "budget_bytes": st.budget,
            "stats": {"puts": st.stats.puts, "gets": st.stats.gets,
                      "gpu": st.stats.hits_gpu, "int8": st.stats.hits_int8,
                      "cpu": st.stats.hits_cpu, "misses": st.stats.misses,
                      "hit_rate": round(st.stats.hit_rate, 3),
                      "demotions": st.stats.demotions}}


def cache_log(n=25):
    return [{"kind": k, "tokens": t, "label": lab}
            for k, t, lab in co.COUNTER.events[-n:]]


def case_file():
    """What the detective is allowed to read. Never secrets, never knows."""
    c = WORLD.case
    ents = c["entities"]
    return {
        "case_id": c["case_id"], "title": c["title"],
        "brief": c["public_brief"].split("You are being questioned")[0].strip(),
        "scene": c.get("crime_scene"), "window": c.get("crime_window", {}),
        "generated_by": c.get("generated_by", "unknown"),
        "known": [{"id": e, "type": ents[e]["type"], "text": ents[e]["text"]}
                  for e in c["public_facts"] if e in ents],
        "suspects": [{"id": s["id"], "name": s["name"], "role": s["role"]}
                     for s in c["suspects"]],
        "board": WORLD.board,
        "discovered": sorted(WORLD.discovered),
    }


def game_json():
    return {
        "difficulty": WORLD.difficulty, "rules": WORLD.rules,
        "turns_used": WORLD.turns_used,
        "turns_left": max(0, WORLD.rules["turns"] - WORLD.turns_used),
        "score": WORLD.score, "finished": WORLD.finished,
        "accusation": WORLD.accusation, "verdict": WORLD.verdict,
        "judge_job": WORLD.judge_job,
        "deepseek": "live" if have_key() else "offline",
        "contradictions_found": len(WORLD.session.hits),
        "contradictions_available": sum(
            len(s.get("lies", [])) for s in WORLD.case["suspects"]),
    }


# ======================================================================
# request models
# ======================================================================

class SuspectReq(BaseModel):
    suspect_id: str

class AskReq(BaseModel):
    suspect_id: str
    question: str
    max_tokens: int = 60

class CommitReq(BaseModel):
    suspect_id: str
    index: int = 0

class RewindReq(BaseModel):
    suspect_id: str
    turn_index: int

class QuoteReq(BaseModel):
    from_suspect: str
    turn_index: int
    to_suspect: str

class ToolReq(BaseModel):
    name: str
    args: dict = {}

class LoadReq(BaseModel):
    case_id: str
    difficulty: str = "normal"

class GenerateReq(BaseModel):
    n_suspects: int = 3
    difficulty: str = "normal"
    hint: str = ""


# ======================================================================
# state
# ======================================================================

@app.get("/api/state")
def get_state():
    return {
        "case": case_file(),
        "game": game_json(),
        "suspects": [suspect_json(s) for s in WORLD.suspects.values()],
        "candidates": {
            sid: [candidate_public(c, i, WORLD.rules["hints"])
                  for i, c in enumerate(cs)]
            for sid, cs in WORLD.candidates.items()},
        "accounting": accounting(), "store": store_json(),
        "log": cache_log(),
    }


@app.get("/api/manual")
def manual():
    if MANUAL.exists():
        return PlainTextResponse(MANUAL.read_text(encoding="utf-8"))
    return PlainTextResponse("MANUAL.md not found next to Api.py")


# ======================================================================
# case library and generation
# ======================================================================

@app.get("/api/cases")
def list_cases():
    return {"cases": library(), "current": WORLD.case["case_id"],
            "difficulties": {k: v["label"] for k, v in DIFFICULTY.items()}}


@app.post("/api/case/load")
def load_case(req: LoadReq):
    WORLD.load(load_case_by_id(req.case_id), req.difficulty)
    return get_state()


@app.post("/api/case/generate")
def generate_case(req: GenerateReq):
    """
    Ask DeepSeek for a new case. Runs in the background; poll /api/job.
    deepseek-reasoner takes one to three minutes per attempt, more if the
    validator sends it back for repairs.
    """
    if not 2 <= req.n_suspects <= 8:
        raise HTTPException(400, "n_suspects must be 2-8")
    difficulty = req.difficulty
    n, hint = req.n_suspects, req.hint

    def work(log):
        client = Deepseek(verbose=False)
        log("offline mode - no DEEPSEEK_API_KEY" if client.offline
            else f"asking {client.model} for a {n}-suspect case")
        gen = CaseGenerator(client, verbose=False)
        seed = int(time.time()) % 100000
        case, report = gen.generate(n_suspects=n, seed=seed, hint=hint)
        log(f"valid after {report['calls']} call(s), "
            f"{report['repairs']} repair(s)")
        CASES_DIR.mkdir(exist_ok=True)
        path = CASES_DIR / f"{case['case_id']}.json"
        path.write_text(json.dumps(case, indent=2), encoding="utf-8")
        log(f"saved {path.name}")
        return {"case_id": case["case_id"], "title": case["title"],
                "generated_by": case["generated_by"], "report": report,
                "difficulty": difficulty}

    return {"job": JOBS.start("generate", work)}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    return JOBS.get(job_id)


# ======================================================================
# play
# ======================================================================

@app.post("/api/explore")
def explore(req: SuspectReq):
    """
    Fork three ways and score them - but hand back only the questions.
    The answers and scores stay server-side until commit. That is the
    game: you choose without seeing what you will get.
    """
    WORLD.require_open()
    s = WORLD.suspect(req.suspect_id)
    cands, cost = Branching.explore(
        s, WORLD.case, WORLD.timeline, WORLD.extractor, WORLD.retriever, n=3)
    WORLD.candidates[s.id] = cands
    return {"suspect_id": s.id,
            "candidates": [candidate_public(c, i, WORLD.rules["hints"])
                           for i, c in enumerate(cands)],
            "cost": cost, "log": cache_log(8)}


@app.post("/api/commit")
def commit(req: CommitReq):
    """Adopt a fork. Now the answer and its score are revealed."""
    WORLD.require_open()
    s = WORLD.suspect(req.suspect_id)
    cands = WORLD.candidates.get(s.id)
    if not cands:
        raise HTTPException(400, "explore first")
    if not 0 <= req.index < len(cands):
        raise HTTPException(400, f"index {req.index} out of range")

    chosen = cands[req.index]
    answer = Branching.commit(s, chosen, WORLD.retriever, WORLD.case)
    turn = s.turns[-1]
    turn.ui_score, turn.ui_band, turn.ui_kind = (
        round(chosen.score, 3), chosen.band(), chosen.kind)

    WORLD.spend_turn()
    outcome = WORLD.record_answer(s, chosen.question, answer, chosen.kind)
    WORLD.candidates.pop(s.id, None)

    return {"answer": answer, "score": round(chosen.score, 3),
            "band": chosen.band(), "kind": chosen.kind,
            "coverage": chosen.coverage, "verifiable": chosen.verifiable,
            **outcome, "suspect": suspect_json(s), "game": game_json()}


@app.post("/api/ask")
def ask(req: AskReq):
    WORLD.require_open()
    s = WORLD.suspect(req.suspect_id)
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "empty question")

    answer, _ = Interrogation.ask_with_composure(
        s, WORLD.session, WORLD.case, WORLD.timeline, WORLD.extractor,
        WORLD.retriever, q, max_tokens=req.max_tokens)
    # ask_with_composure already ran check_answer; undo the double count
    # by scoring here from the recorded state instead
    turn = s.turns[-1]
    hit = any(h.turn_index == turn.index for h in WORLD.session.hits_for(s.id))
    mentioned = WORLD.extractor.extract(answer)
    new = WORLD.discover(mentioned, "statement", f"{s.name} said it",
                         s.id, turn.index)
    WORLD.spend_turn()
    if hit:
        WORLD.score += SCORE["contradiction"]
        note = next(h.note for h in WORLD.session.hits_for(s.id)
                    if h.turn_index == turn.index)
        WORLD.pin(s.id, "contradiction", note, turn.index)
    else:
        note = None
    if new == 0 and not hit:
        WORLD.score += SCORE["wasted"]

    WORLD.candidates.pop(s.id, None)
    return {"answer": answer, "contradiction": note, "new_facts": new,
            "violations": sorted(WORLD.checker.violations(answer, s.knows)),
            "suspect": suspect_json(s), "game": game_json()}


@app.post("/api/rewind")
def rewind(req: RewindReq):
    """Crop back. Costs turns on harder difficulties."""
    if WORLD.finished:
        raise HTTPException(400, "the case is closed")
    s = WORLD.suspect(req.suspect_id)
    if not 0 <= req.turn_index < len(s.turns):
        raise HTTPException(400, f"no turn {req.turn_index}")

    dropped = len(s.turns) - req.turn_index
    tier = s.restore_snapshot(req.turn_index)

    # forget what those turns taught the detective, too
    WORLD.board = [b for b in WORLD.board
                   if not (b.get("suspect") == s.id and b.get("turn") is not None
                           and b["turn"] >= req.turn_index)]
    WORLD.discovered = set(WORLD.case["public_facts"]) | {
        b["entity"] for b in WORLD.board}
    WORLD.pins[s.id] = [p for p in WORLD.pins[s.id]
                        if p.get("turn") is None or p["turn"] < req.turn_index]
    WORLD.session.hits = [h for h in WORLD.session.hits
                          if not (h.suspect_id == s.id
                                  and h.turn_index >= req.turn_index)]

    cost = WORLD.rules["rewind_cost"]
    if cost == 0:
        WORLD.turns_used = max(0, WORLD.turns_used - dropped)   # refund
    else:
        WORLD.turns_used = min(WORLD.rules["turns"], WORLD.turns_used + cost)
        WORLD.score += SCORE["turn"] * cost

    WORLD.candidates.pop(s.id, None)
    return {"tier": tier, "dropped": dropped, "cost": cost,
            "suspect": suspect_json(s), "game": game_json(),
            "store": store_json(), "log": cache_log(8)}


@app.post("/api/quote")
def quote(req: QuoteReq):
    WORLD.require_open()
    src, dst = WORLD.suspect(req.from_suspect), WORLD.suspect(req.to_suspect)
    if not 0 <= req.turn_index < len(src.turns):
        raise HTTPException(400, f"{src.id} has no turn {req.turn_index}")

    question, granted = WORLD.session.quote(
        src, req.turn_index, dst, WORLD.case, WORLD.extractor)
    answer, note = Interrogation.ask_with_composure(
        dst, WORLD.session, WORLD.case, WORLD.timeline, WORLD.extractor,
        WORLD.retriever, question, granted=granted)

    turn = dst.turns[-1]
    WORLD.spend_turn()
    new = WORLD.discover(WORLD.extractor.extract(answer), "statement",
                         f"{dst.name} said it", dst.id, turn.index)
    if note:
        WORLD.score += SCORE["contradiction"]
        WORLD.pin(dst.id, "contradiction", note, turn.index)
    if granted:
        WORLD.pin(dst.id, "quoted",
                  f"told what {src.name} said; now knows "
                  + ", ".join(granted), turn.index)
    WORLD.candidates.pop(dst.id, None)
    return {"question": question, "answer": answer, "granted": granted,
            "contradiction": note, "new_facts": new,
            "suspect": suspect_json(dst), "game": game_json()}


# ======================================================================
# forensics (MCP)
# ======================================================================

@app.get("/api/tools")
def list_tools():
    try:
        return {"tools": [{"name": n, "doc": d}
                          for n, d in WORLD.tools().available().items()]}
    except Exception as exc:
        raise HTTPException(503, f"forensic server unreachable: {exc}")


@app.post("/api/tool")
def call_tool(req: ToolReq):
    """
    Call a forensic tool over MCP and put anything it reveals on the
    evidence board. Costs a turn: the lab does not work for free.
    """
    WORLD.require_open()
    try:
        f = WORLD.tools()
        result = f.client.call(f._resolve(req.name), **req.args)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"tool call failed: {exc}")

    WORLD.spend_turn()
    found = set()
    text = json.dumps(result)
    for eid in WORLD.case["entities"]:
        if eid in text:
            found.add(eid)
    new = WORLD.discover(found, "forensics", req.name.replace("_tool", ""))

    sid = req.args.get("suspect_id")
    note = None
    if isinstance(result, dict):
        if result.get("verdict") == "contradicted" and sid:
            note = result.get("detail")
        elif result.get("contradicted") and sid:
            bad = [c for c in result.get("checks", [])
                   if c.get("verdict") == "contradicted"]
            note = bad[0]["detail"] if bad else "statement contradicted"
    if note and sid in WORLD.pins:
        WORLD.pin(sid, "forensics", note)
        WORLD.score += SCORE["contradiction"] // 2

    return {"result": result, "new_facts": new, "flag": note,
            "game": game_json(), "board": WORLD.board}


# ======================================================================
# accuse -> judge
# ======================================================================

@app.post("/api/accuse")
def accuse(req: SuspectReq):
    """
    Close the case and send the transcript plus the truth to the judge.
    The judge runs in the background (deepseek-reasoner is slow); the
    verdict lands in state.game.verdict when it is ready.
    """
    if WORLD.finished:
        raise HTTPException(400, "already accused")
    s = WORLD.suspect(req.suspect_id)
    WORLD.finished = True
    WORLD.accusation = s.id
    truth = WORLD.case["culprit"]

    case, suspects, session = WORLD.case, WORLD.suspects, WORLD.session

    def work(log):
        client = Deepseek(verbose=False)
        log("offline judge - no DEEPSEEK_API_KEY" if client.offline
            else f"judging with {client.model}")
        v = CaseJudge(client, verbose=False).judge(case, suspects, s.id, session)
        d = v.as_dict()
        if v.correct and v.evidence_backed:
            d["points"] = SCORE["correct_backed"]
        elif v.correct:
            d["points"] = SCORE["correct_lucky"]
        else:
            d["points"] = SCORE["wrong"]
        WORLD.score += d["points"]
        d["final_score"] = WORLD.score
        d["accused_name"] = s.name
        d["culprit_name"] = WORLD.suspects[truth].name
        WORLD.verdict = d
        return d

    WORLD.judge_job = JOBS.start("judge", work)
    return {"accused": s.id, "accused_name": s.name,
            "job": WORLD.judge_job, "game": game_json()}


@app.post("/api/reset")
def reset():
    WORLD.load(WORLD.case, WORLD.difficulty)
    return get_state()


# ======================================================================
# frontend
# ======================================================================

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"),
              name="assets")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    def index_missing():
        return {"message": "frontend not built: cd frontend && npm install && npm run build"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1, log_level="info")