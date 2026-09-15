"""
Test_Api.py   -  Stage 9 acceptance test

Drives every endpoint against a real FastAPI app with the real model
loaded, using Starlette's TestClient (no network, but the same code path
the browser hits).

What this proves beyond "the routes return 200":
  - the cache persists across requests, which is the whole reason the
    backend has to be one long-lived process
  - exploring costs far less prefill than three naive trials
  - rewinding through the API restores the suspect and reports which
    storage tier the snapshot came back from
  - quoting grants knowledge and rewinding takes it away
  - the frontend build is present and mounted

Run:  py Test_Api.py
"""

from pathlib import Path

from fastapi.testclient import TestClient

import Api


def show(title):
    print(f"\n{title}")


def run():
    # TestClient runs the lifespan handler, so the model loads here
    with TestClient(Api.app) as client:

        # ---------------------------------------------------- state
        show("test_state_shape")
        st = client.get("/api/state").json()
        assert len(st["suspects"]) == 3, st["suspects"]
        assert st["accounting"]["shared_tokens"] > 0
        assert st["store"]["budget_bytes"] > 0
        ids = [s["id"] for s in st["suspects"]]
        print(f"  {len(ids)} suspects, {st['accounting']['shared_tokens']} shared "
              f"tokens, {st['accounting']['bytes_per_token']} B/token")
        print(f"  store: {st['store']['report']}")

        sid = "vance"

        # ---------------------------------------------------- explore
        show("test_explore_is_cheap")
        before = client.get("/api/state").json()["accounting"]["prefill_tokens"]
        r = client.post("/api/explore", json={"suspect_id": sid})
        assert r.status_code == 200, r.text
        exp = r.json()
        after = client.get("/api/state").json()["accounting"]["prefill_tokens"]

        assert len(exp["candidates"]) == 3, exp
        assert exp["cost"]["forks"] == 3, exp["cost"]
        spent = after - before
        suspect = next(s for s in client.get("/api/state").json()["suspects"]
                       if s["id"] == sid)
        naive = 3 * (suspect["cache_tokens"] + 20)
        print(f"  3 forks cost {spent} prefill tokens; "
              f"3 naive trials would cost ~{naive} "
              f"({100 * (1 - spent / naive):.0f}% cheaper)")
        assert spent < naive / 2

        for c in exp["candidates"]:
            assert 0.0 <= c["score"] <= 1.0
            assert c["band"] in ("low", "medium", "high")
        print(f"  bands {[c['band'] for c in exp['candidates']]}")

        # ---------------------------------------------------- commit
        show("test_commit_uses_the_previewed_answer")
        preview = exp["candidates"][0]["preview"]
        r = client.post("/api/commit", json={"suspect_id": sid, "index": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer"] == preview, "committed answer differs from preview"
        assert len(body["suspect"]["turns"]) == 1
        print(f"  committed: {body['answer'][:58]}")

        # ---------------------------------------------------- persistence
        show("test_cache_persists_between_requests")
        a = client.get("/api/state").json()
        s_a = next(s for s in a["suspects"] if s["id"] == sid)
        client.get("/api/state")
        b = client.get("/api/state").json()
        s_b = next(s for s in b["suspects"] if s["id"] == sid)
        assert s_a["cache_tokens"] == s_b["cache_tokens"] > s_a["base_tokens"]
        print(f"  cache held at {s_b['cache_tokens']} tokens across requests "
              f"(base {s_b['base_tokens']})")

        # ---------------------------------------------------- free-text ask
        show("test_ask")
        r = client.post("/api/ask", json={
            "suspect_id": sid, "question": "Did you go up the stairs?",
            "max_tokens": 25})
        assert r.status_code == 200, r.text
        assert len(r.json()["suspect"]["turns"]) == 2
        print(f"  answer: {r.json()['answer'][:58]}")

        # ---------------------------------------------------- rewind
        show("test_rewind_reports_its_tier")
        r = client.post("/api/rewind", json={"suspect_id": sid, "turn_index": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["tier"] in ("gpu", "int8", "cpu", "recomputed"), body["tier"]
        assert len(body["suspect"]["turns"]) == 0
        assert body["suspect"]["cache_tokens"] == body["suspect"]["base_tokens"]
        print(f"  restored from '{body['tier']}', back to "
              f"{body['suspect']['cache_tokens']} tokens")
        print(f"  store now: {body['store']['report']}")

        # ---------------------------------------------------- quoting
        show("test_quote_grants_then_rewind_removes")
        client.post("/api/ask", json={
            "suspect_id": "vance",
            "question": "Where were you at 21:38?", "max_tokens": 25})

        # Pin what Vance said. A 0.5B model may or may not name a time on
        # its own, and what is under test here is the grant/rewind
        # mechanism, not the model's wording.
        Api.WORLD.suspects["vance"].turns[0].answer = (
            "I went up to the observatory at 21:38 and left at 21:46.")

        before_knows = set(next(
            s for s in client.get("/api/state").json()["suspects"]
            if s["id"] == "rourke")["knows"])

        r = client.post("/api/quote", json={
            "from_suspect": "vance", "turn_index": 0, "to_suspect": "rourke"})
        assert r.status_code == 200, r.text
        granted = r.json()["granted"]
        after_knows = set(r.json()["suspect"]["knows"])
        assert after_knows >= before_knows
        assert granted, "pinned answer should have surfaced new entities"
        assert set(granted) <= after_knows
        print(f"  granted {granted}")

        r = client.post("/api/rewind",
                        json={"suspect_id": "rourke", "turn_index": 0})
        back = set(r.json()["suspect"]["knows"])
        assert back == before_knows, (back, before_knows)
        for eid in granted:
            assert eid not in back, f"{eid} survived the rewind"
        print("  rewind removed every granted fact")

        # ---------------------------------------------------- MCP tools
        show("test_forensic_tools_over_mcp")
        r = client.get("/api/tools")
        if r.status_code == 200:
            names = [t["name"] for t in r.json()["tools"]]
            assert names, "no tools advertised"
            r2 = client.post("/api/tool", json={
                "name": "check_alibi",
                "args": {"suspect_id": "vance", "place": "observatory",
                         "at": "21:40"}})
            assert r2.status_code == 200, r2.text
            assert r2.json()["result"]["verdict"] == "supported"
            print(f"  {len(names)} tools discovered; check_alibi -> supported")
        else:
            print(f"  SKIPPED - forensic server unreachable ({r.status_code})")

        # ---------------------------------------------------- accuse
        show("test_accuse")
        r = client.post("/api/accuse", json={"suspect_id": "vance"})
        assert r.status_code == 200, r.text
        v = r.json()
        assert v["culprit"] == "vance" and v["correct"] is True
        print(f"  accused {v['accused_name']} -> "
              f"{'CORRECT' if v['correct'] else 'WRONG'}, "
              f"{v['contradictions_found']} contradictions found")

        # ---------------------------------------------------- reset
        show("test_reset")
        st = client.post("/api/reset", json={}).json()
        assert all(len(s["turns"]) == 0 for s in st["suspects"])
        assert st["finished"] is False
        print("  all suspects back to base")

        # ---------------------------------------------------- errors
        show("test_bad_requests_are_rejected")
        assert client.post("/api/explore",
                           json={"suspect_id": "nobody"}).status_code == 404
        assert client.post("/api/commit",
                           json={"suspect_id": "vance", "index": 0}).status_code == 400
        assert client.post("/api/rewind",
                           json={"suspect_id": "vance",
                                 "turn_index": 9}).status_code == 400
        print("  unknown suspect 404, uncommitted 400, bad turn 400")

        # ---------------------------------------------------- frontend
        show("test_frontend_is_built_and_served")
        dist = Path(Api.FRONTEND_DIST)
        if dist.exists():
            assert (dist / "index.html").exists()
            r = client.get("/")
            assert r.status_code == 200
            print(f"  dist present, / serves index.html "
                  f"({len(r.content)} bytes)")
        else:
            print("  SKIPPED - run: cd frontend && npm install && npm run build")


if __name__ == "__main__":
    print("=" * 62)
    print("STAGE 9  API")
    print("=" * 62)
    run()
    print("\n" + "=" * 62)
    print("stage 9 PASSED")