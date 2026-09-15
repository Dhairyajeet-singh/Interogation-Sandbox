"""
Api.py   -  Stage 9
-------------------
FastAPI backend. All the logic lives here; the React app only renders.

ONE PROCESS, ONE WORKER, NO RELOAD
----------------------------------
The KV caches live in this process's memory. Run more than one worker and
requests land in different processes with different caches. Run with
--reload and every file save wipes the model. So:

    py Api.py                       correct
    uvicorn Api:app --reload        WRONG - destroys the cache on save
    uvicorn Api:app --workers 4     WRONG - each worker has its own caches

The model is loaded once in the lifespan handler and held for the life of
the server. That is also why this cannot be deployed to anything that
scales to zero.

Endpoints
    GET  /api/state                 everything the UI draws
    POST /api/explore               stage 5: fork 3 ways, score, return options
    POST /api/commit                adopt one of those forks
    POST /api/ask                   free-text question
    POST /api/rewind                crop back to a turn
    POST /api/quote                 tell A what B said
    POST /api/tool                  call a forensic tool over MCP
    POST /api/accuse                end the game
    POST /api/reset                 start over
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import Branching
import cache_ops as co
import Interrogation
from Extractor import FactExtractor
from Model_setup import load_model
from Retrieval import HybridRetriever, Timeline, KnowledgeChecker
from Snapshots import SnapshotStore
from Suspects import (Suspect, compute_shared_len, make_shared_block,
                      make_private_block)

HERE = Path(__file__).parent
CASE_PATH = HERE / "Case_Ashfield.json"
FRONTEND_DIST = HERE / "frontend" / "dist"

# GPU budget for snapshots. Small on purpose so eviction actually happens
# and the UI has something to show.
SNAPSHOT_BUDGET_MB = 96


# ======================================================================
# the world
# ======================================================================

class World:
    """Model, suspects, caches. Built once, held for the life of the process."""

    def __init__(self):
        self.case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
        self.tok, self.model, self.device = load_model()

        self.extractor = FactExtractor(self.case["entities"])
        self.timeline = Timeline(self.case["timeline"])
        self.retriever = HybridRetriever(self.case["entities"], self.timeline)
        self.checker = KnowledgeChecker(self.extractor, self.case["public_facts"])

        self.store = SnapshotStore(
            gpu_budget_bytes=SNAPSHOT_BUDGET_MB * 1024 * 1024,
            device=self.device,
        )

        shared_block = make_shared_block(self.case)
        privates = [make_private_block(self.case, sp)
                    for sp in self.case["suspects"]]
        self.n_shared, self.shared_ids = compute_shared_len(
            self.tok, shared_block, privates)
        self.shared_cache, _ = co.prefill(
            self.model,
            torch.tensor([self.shared_ids], device=self.device),
            label="shared",
        )

        self.suspects = {}
        self.session = Interrogation.Session()
        self.candidates = {}          # suspect_id -> last explored candidates
        self.finished = False
        self.accusation = None
        self._build_suspects()

        self.forensics = None         # connected lazily; MCP spawn is slow

    def _build_suspects(self):
        self.suspects = {
            spec["id"]: Suspect(spec, self.case, self.shared_cache,
                                self.shared_ids, self.model, self.tok,
                                self.device, store=self.store)
            for spec in self.case["suspects"]
        }

    def tools(self):
        if self.forensics is None:
            from Mcp_client import ForensicsClient, Forensics
            self.forensics = Forensics(ForensicsClient())
        return self.forensics

    def suspect(self, sid):
        if sid not in self.suspects:
            raise HTTPException(404, f"no suspect {sid!r}")
        return self.suspects[sid]

    def reset(self):
        self.store = SnapshotStore(
            gpu_budget_bytes=SNAPSHOT_BUDGET_MB * 1024 * 1024,
            device=self.device)
        self.session = Interrogation.Session()
        self.candidates = {}
        self.finished = False
        self.accusation = None
        co.COUNTER.reset()
        self._build_suspects()


WORLD: World | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global WORLD
    WORLD = World()
    print(f"ready: {len(WORLD.suspects)} suspects, "
          f"{WORLD.n_shared} shared tokens on {WORLD.device}")
    yield


app = FastAPI(title="Branching Interrogation Sandbox", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)


# ======================================================================
# serialisation
# ======================================================================

def suspect_json(s):
    return {
        "id": s.id,
        "name": s.name,
        "role": s.role,
        "composure": round(s.composure, 3),
        "temperature": Interrogation.temperature_for(s.composure),
        "cache_tokens": co.length(s.cache),
        "base_tokens": s.base_len,
        "knows": s.knows,
        "turns": [
            {
                "index": t.index,
                "question": t.question,
                "answer": t.answer,
                "cache_len_before": t.cache_len_before,
                "retrieved": t.retrieved,
                "granted": t.granted,
                "snapshot_tier": (
                    s.store.entries[t.snapshot_key].tier
                    if t.snapshot_key in s.store.entries else "dropped"
                ),
            }
            for t in s.turns
        ],
        "hits": [
            {"turn": h.turn_index, "note": h.note}
            for h in WORLD.session.hits_for(s.id)
        ],
    }


def candidate_json(c, index):
    return {
        "index": index,
        "kind": c.kind,
        "question": c.question,
        "preview": c.answer,
        "score": round(c.score, 3),
        "band": c.band(),
        "coverage": c.coverage,
        "verifiable": c.verifiable,
        "contradiction": c.contradiction,
        "contradiction_note": c.contradiction_note,
    }


def accounting():
    """
    What the cache bought us.

    naive = every suspect reads the shared block itself.
    real  = it was read once and forked.
    """
    n = len(WORLD.suspects)
    saved = WORLD.n_shared * (n - 1)
    return {
        "prefill_tokens": co.COUNTER.prefill_tokens,
        "decode_tokens": co.COUNTER.decode_tokens,
        "forks": co.COUNTER.count("fork"),
        "crops": co.COUNTER.count("crop"),
        "evictions": co.COUNTER.count("evict"),
        "shared_tokens": WORLD.n_shared,
        "suspects": n,
        "tokens_saved_on_shared_block": saved,
        "bytes_per_token": round(co.bytes_per_token(WORLD.shared_cache)),
    }


def store_json():
    st = WORLD.store
    return {
        "report": st.report(),
        "tiers": st.tier_counts(),
        "gpu_bytes": st.gpu_bytes(),
        "budget_bytes": st.budget,
        "stats": {
            "puts": st.stats.puts, "gets": st.stats.gets,
            "gpu": st.stats.hits_gpu, "int8": st.stats.hits_int8,
            "cpu": st.stats.hits_cpu, "misses": st.stats.misses,
            "hit_rate": round(st.stats.hit_rate, 3),
            "demotions": st.stats.demotions,
        },
    }


def cache_log(n=25):
    rows = co.COUNTER.events[-n:]
    return [{"kind": k, "tokens": t, "label": lab} for k, t, lab in rows]


# ======================================================================
# requests
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
    use_snapshot: bool = True


class QuoteReq(BaseModel):
    from_suspect: str
    turn_index: int
    to_suspect: str


class ToolReq(BaseModel):
    name: str
    args: dict = {}


# ======================================================================
# endpoints
# ======================================================================

@app.get("/api/state")
def get_state():
    return {
        "case": {
            "title": WORLD.case["title"],
            "brief": WORLD.case["public_brief"],
            # the UI needs a real place id for the forensic tools; without
            # this it would have to hard-code one, which breaks on any
            # generated case
            "scene": WORLD.case.get("crime_scene") or next(
                (e for e, v in WORLD.case["entities"].items()
                 if v["type"] == "place"), None),
            "window": WORLD.case.get("crime_window", {}),
        },
        "suspects": [suspect_json(s) for s in WORLD.suspects.values()],
        "candidates": {
            sid: [candidate_json(c, i) for i, c in enumerate(cs)]
            for sid, cs in WORLD.candidates.items()
        },
        "accounting": accounting(),
        "store": store_json(),
        "log": cache_log(),
        "finished": WORLD.finished,
        "accusation": WORLD.accusation,
    }


@app.post("/api/explore")
def explore(req: SuspectReq):
    """Stage 5: fork three ways, try a different question in each, score."""
    s = WORLD.suspect(req.suspect_id)
    cands, cost = Branching.explore(
        s, WORLD.case, WORLD.timeline, WORLD.extractor, WORLD.retriever, n=3)
    WORLD.candidates[s.id] = cands
    return {
        "suspect_id": s.id,
        "candidates": [candidate_json(c, i) for i, c in enumerate(cands)],
        "cost": cost,
        "log": cache_log(8),
    }


@app.post("/api/commit")
def commit(req: CommitReq):
    """Adopt one explored fork as the suspect's real next turn."""
    s = WORLD.suspect(req.suspect_id)
    cands = WORLD.candidates.get(s.id)
    if not cands:
        raise HTTPException(400, "nothing explored for this suspect yet")
    if not 0 <= req.index < len(cands):
        raise HTTPException(400, f"index {req.index} out of range")

    chosen = cands[req.index]
    answer = Branching.commit(s, chosen, WORLD.retriever, WORLD.case)

    note = WORLD.session.check_answer(
        s, WORLD.case, WORLD.timeline, WORLD.extractor,
        chosen.question, answer, s.turns[-1].index)
    if chosen.kind == "confront_evidence":
        WORLD.session.apply_pressure(s)

    WORLD.candidates.pop(s.id, None)
    return {
        "answer": answer,
        "contradiction": note,
        "violations": sorted(WORLD.checker.violations(answer, s.knows)),
        "suspect": suspect_json(s),
    }


@app.post("/api/ask")
def ask(req: AskReq):
    """A question the player typed themselves."""
    s = WORLD.suspect(req.suspect_id)
    answer, note = Interrogation.ask_with_composure(
        s, WORLD.session, WORLD.case, WORLD.timeline, WORLD.extractor,
        WORLD.retriever, req.question, max_tokens=req.max_tokens)
    WORLD.candidates.pop(s.id, None)
    return {
        "answer": answer,
        "contradiction": note,
        "violations": sorted(WORLD.checker.violations(answer, s.knows)),
        "suspect": suspect_json(s),
    }


@app.post("/api/rewind")
def rewind(req: RewindReq):
    """
    Crop the suspect's cache back to before a turn.

    use_snapshot=True goes through the store, so the response reports
    which tier it came back from - that is the interesting part.
    """
    s = WORLD.suspect(req.suspect_id)
    if not 0 <= req.turn_index < len(s.turns):
        raise HTTPException(400, f"no turn {req.turn_index}")

    if req.use_snapshot:
        tier = s.restore_snapshot(req.turn_index)
    else:
        s.rewind_to(req.turn_index)
        tier = "crop"

    WORLD.candidates.pop(s.id, None)
    return {"tier": tier, "suspect": suspect_json(s),
            "store": store_json(), "log": cache_log(8)}


@app.post("/api/quote")
def quote(req: QuoteReq):
    """Tell one suspect what another said. Grants knowledge; rewind undoes it."""
    src = WORLD.suspect(req.from_suspect)
    dst = WORLD.suspect(req.to_suspect)
    if not 0 <= req.turn_index < len(src.turns):
        raise HTTPException(400, f"{src.id} has no turn {req.turn_index}")

    question, granted = WORLD.session.quote(
        src, req.turn_index, dst, WORLD.case, WORLD.extractor)
    answer, note = Interrogation.ask_with_composure(
        dst, WORLD.session, WORLD.case, WORLD.timeline, WORLD.extractor,
        WORLD.retriever, question, granted=granted)

    WORLD.candidates.pop(dst.id, None)
    return {"question": question, "answer": answer, "granted": granted,
            "contradiction": note, "suspect": suspect_json(dst)}


@app.get("/api/tools")
def list_tools():
    """What the MCP forensic server advertises. Discovered, not hard-coded."""
    try:
        return {"tools": [{"name": n, "doc": d}
                          for n, d in WORLD.tools().available().items()]}
    except Exception as exc:
        raise HTTPException(503, f"forensic server unreachable: {exc}")


@app.post("/api/tool")
def call_tool(req: ToolReq):
    """Invoke a forensic tool over MCP."""
    try:
        f = WORLD.tools()
        return {"result": f.client.call(f._resolve(req.name), **req.args)}
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"tool call failed: {exc}")


@app.post("/api/accuse")
def accuse(req: SuspectReq):
    s = WORLD.suspect(req.suspect_id)
    truth = WORLD.case["culprit"]
    WORLD.finished = True
    WORLD.accusation = s.id

    available = sum(len(sp.get("lies", [])) for sp in WORLD.case["suspects"])
    return {
        "accused": s.id,
        "accused_name": s.name,
        "culprit": truth,
        "culprit_name": WORLD.suspects[truth].name,
        "correct": s.id == truth,
        "contradictions_found": len(WORLD.session.hits),
        "contradictions_available": available,
        "accounting": accounting(),
    }


@app.post("/api/reset")
def reset():
    WORLD.reset()
    return get_state()


# ======================================================================
# serve the built frontend, if there is one
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
        return {"message": "frontend not built. cd frontend && npm install "
                           "&& npm run build, or run the Vite dev server."}


if __name__ == "__main__":
    import uvicorn
    # one worker, no reload - see the note at the top of this file
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1, log_level="info")