"""
Mcp_server.py   -  Stage 8
--------------------------
The forensic tools, exposed over the Model Context Protocol.

WHY MCP RATHER THAN JUST CALLING THE FUNCTIONS
----------------------------------------------
The timeline checker and records lookup are a genuinely separate service.
They reason over the case record, not over any suspect's memory, and they
are the only part of the system the detective can consult that the
suspects cannot influence. Running them behind MCP means:

  - they live in their own process, so a crash there cannot corrupt the
    KV caches in the main one
  - the same server can be pointed at by any MCP client, including
    Claude Desktop or Cursor, not only by this game
  - adding a tool needs no change to the agent

Two transports, one server:

    py Mcp_server.py                 stdio  (what an MCP client spawns)
    py Mcp_server.py --http --port 8765    HTTP (easy to curl and debug)

The SDK renamed its server class in 2.x, so there is a small compat shim
at the top. Same idea as the cache format shim in cache_ops.
"""

import json
import os
import sys
from pathlib import Path

from Extractor import FactExtractor
from Retrieval import Timeline

# Which case these tools reason over.
#
# This USED to be hardcoded to Case_Ashfield.json, which meant that as
# soon as you played a generated case the lab was answering from the
# wrong case file - movements_of returned an empty list because the
# suspect did not exist in Ashfield. The path now comes from the client
# that spawned us, so the lab always matches the game.
def _case_path():
    if "--case" in sys.argv:
        return Path(sys.argv[sys.argv.index("--case") + 1])
    env = os.environ.get("SANDBOX_CASE_PATH")
    if env:
        return Path(env)
    return Path(__file__).with_name("Case_Ashfield.json")


CASE_PATH = _case_path()


# ======================================================================
# the case record these tools reason over
# ======================================================================

def load_case(path=CASE_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


CASE = load_case()
TIMELINE = Timeline(CASE["timeline"])
EXTRACTOR = FactExtractor(CASE["entities"])


def _actor_id(suspect_id):
    """Accept 'vance' or 'per_vance' and return the graph's node id."""
    return suspect_id if suspect_id.startswith("per_") else f"per_{suspect_id}"


def _place_id(place):
    """Accept 'observatory', 'loc_observatory', or any alias."""
    if place.startswith("loc_"):
        return place
    low = place.lower().strip()
    for eid, ent in CASE["entities"].items():
        if ent["type"] != "place":
            continue
        if eid == f"loc_{low}" or any(a.lower() == low for a in ent["aliases"]):
            return eid
    return f"loc_{low}"


# ======================================================================
# tool implementations - plain functions, easy to test without MCP
# ======================================================================

def check_alibi(suspect_id: str, place: str, at: str) -> dict:
    """
    Does the case record put this person in this place at this time?

    at is 24-hour HH:MM, e.g. "21:40".
    """
    actor, loc = _actor_id(suspect_id), _place_id(place)
    verdict, why = TIMELINE.check_alibi(actor, loc, at)
    return {"suspect": actor, "place": loc, "at": at,
            "verdict": verdict, "detail": why}


def who_was_at(place: str, at: str = None) -> dict:
    """Everyone the record places at a location, optionally at one time."""
    loc = _place_id(place)
    rows = TIMELINE.who_was_at(loc, at=at)
    return {"place": loc, "at": at,
            "people": [{"suspect": r["actor"], "from": r["from"],
                        "to": r["to"], "verified_by": r["verifiable_by"]}
                       for r in rows]}


def movements_of(suspect_id: str) -> dict:
    """The full recorded movement history for one person."""
    actor = _actor_id(suspect_id)
    rows = TIMELINE.where_was(actor)
    return {"suspect": actor,
            "movements": [{"place": r["place"], "from": r["from"],
                           "to": r["to"], "verified_by": r["verifiable_by"]}
                          for r in rows]}


def lookup_evidence(query: str) -> dict:
    """Find a piece of evidence by id, name or alias."""
    low = query.lower().strip()
    for eid, ent in CASE["entities"].items():
        if eid == low or any(a.lower() == low for a in ent["aliases"]):
            return {"id": eid, "type": ent["type"], "text": ent["text"],
                    "aliases": ent["aliases"]}
    matches = [eid for eid, ent in CASE["entities"].items()
               if low in ent["text"].lower()]
    if matches:
        eid = matches[0]
        return {"id": eid, "type": CASE["entities"][eid]["type"],
                "text": CASE["entities"][eid]["text"],
                "aliases": CASE["entities"][eid]["aliases"]}
    return {"error": f"nothing matching {query!r}",
            "known_ids": sorted(CASE["entities"])[:12]}


def verify_statement(suspect_id: str, statement: str) -> dict:
    """
    Pull every checkable claim out of a statement and adjudicate each one
    against the timeline. This is the tool the detective reaches for after
    a suspect says something specific.
    """
    found = EXTRACTOR.extract(statement)
    ents = CASE["entities"]
    places = [e for e in found if ents[e]["type"] == "place"]
    times = [e for e in found if ents[e]["type"] == "time"]

    checks = []
    for p in places:
        for t in times:
            clock = None
            for alias in ents[t]["aliases"]:
                if ":" in alias and alias.replace(":", "").isdigit():
                    h, m = alias.split(":")
                    clock = f"{int(h):02d}:{m}"
                    break
            if clock is None:
                continue
            verdict, why = TIMELINE.check_alibi(_actor_id(suspect_id), p, clock)
            checks.append({"place": p, "at": clock,
                           "verdict": verdict, "detail": why})

    return {"suspect": _actor_id(suspect_id),
            "entities_mentioned": sorted(found),
            "checks": checks,
            "contradicted": any(c["verdict"] == "contradicted" for c in checks)}


def case_timeline() -> str:
    """The whole recorded timeline, as readable lines."""
    lines = []
    for e in CASE["timeline"]:
        line = f"{e['actor']} at {e['place']} {e['from']}-{e['to']}"
        if e["verifiable_by"]:
            line += f" (verified by {e['verifiable_by']})"
        lines.append(line)
    return "\n".join(lines)


def case_evidence() -> str:
    """Every evidence item in the case record."""
    return "\n".join(
        f"{eid}: {ent['text']}"
        for eid, ent in CASE["entities"].items()
        if ent["type"] == "evidence"
    )


TOOLS = {
    "check_alibi": check_alibi,
    "who_was_at": who_was_at,
    "movements_of": movements_of,
    "lookup_evidence": lookup_evidence,
    "verify_statement": verify_statement,
}

RESOURCES = {
    "case://timeline": case_timeline,
    "case://evidence": case_evidence,
}


# ======================================================================
# MCP server - compat shim for SDK v1 and v2
# ======================================================================

def _server_class():
    """
    mcp 1.x calls it FastMCP; mcp 2.x renamed it to MCPServer.
    Same decorator API either way.
    """
    try:
        from mcp.server.mcpserver import MCPServer      # 2.x
        return MCPServer
    except ImportError:
        pass
    from mcp.server.fastmcp import FastMCP              # 1.x
    return FastMCP


def build_server(name=None):
    name = name or f"forensics-{CASE['case_id']}"
    Server = _server_class()
    try:
        app = Server(name, instructions=(
            f"Forensic tools for {CASE['title']!r}. Use these to check a "
            f"suspect's claims against the physical record."
        ))
    except TypeError:
        app = Server(name)

    @app.tool()
    def check_alibi_tool(suspect_id: str, place: str, at: str) -> dict:
        """Check whether the record puts a suspect somewhere at a given time."""
        return check_alibi(suspect_id, place, at)

    @app.tool()
    def who_was_at_tool(place: str, at: str = None) -> dict:
        """List everyone recorded at a place, optionally at one time."""
        return who_was_at(place, at)

    @app.tool()
    def movements_of_tool(suspect_id: str) -> dict:
        """Full recorded movements for one suspect."""
        return movements_of(suspect_id)

    @app.tool()
    def lookup_evidence_tool(query: str) -> dict:
        """Look up a piece of evidence by id, name or alias."""
        return lookup_evidence(query)

    @app.tool()
    def verify_statement_tool(suspect_id: str, statement: str) -> dict:
        """Adjudicate every checkable claim in a statement."""
        return verify_statement(suspect_id, statement)

    @app.resource("case://timeline")
    def timeline_resource() -> str:
        """The recorded movements of everyone in the case."""
        return case_timeline()

    @app.resource("case://evidence")
    def evidence_resource() -> str:
        """Every evidence item in the case record."""
        return case_evidence()

    return app


# ======================================================================
# HTTP transport - a thin JSON wrapper over the same functions
# ======================================================================

def build_http_app():
    """
    FastAPI mirror of the same tools, for debugging and for the browser.

    Deliberately the same functions, so there is no chance of the two
    transports drifting apart.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    api = FastAPI(title="Ashfield forensics (HTTP transport)")

    class Call(BaseModel):
        args: dict = {}

    @api.get("/tools")
    def list_tools():
        return {"tools": [
            {"name": n, "doc": (f.__doc__ or "").strip()}
            for n, f in TOOLS.items()
        ], "resources": sorted(RESOURCES)}

    @api.post("/tools/{name}")
    def call_tool(name: str, body: Call):
        if name not in TOOLS:
            raise HTTPException(404, f"no tool named {name!r}")
        try:
            return {"ok": True, "result": TOOLS[name](**body.args)}
        except TypeError as exc:
            raise HTTPException(400, str(exc))

    @api.get("/resources")
    def read_resource(uri: str):
        if uri not in RESOURCES:
            raise HTTPException(404, f"no resource {uri!r}")
        return {"uri": uri, "text": RESOURCES[uri]()}

    return api


# ======================================================================

def main():
    if "--http" in sys.argv:
        import uvicorn
        port = 8765
        if "--port" in sys.argv:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        print(f"forensics HTTP transport on http://127.0.0.1:{port} "
              f"serving {CASE['case_id']}", file=sys.stderr)
        uvicorn.run(build_http_app(), host="127.0.0.1", port=port,
                    log_level="warning")
    else:
        build_server().run()          # stdio, the default MCP transport


if __name__ == "__main__":
    main()