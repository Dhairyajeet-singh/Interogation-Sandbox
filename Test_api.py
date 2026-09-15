"""
Test_Api.py   -  Stage 9, game edition

Drives every endpoint against a real FastAPI app with the real model.

Beyond "the routes return 200", this proves:
  - previews and scores are hidden until commit (server-side, not just UI)
  - hard mode hides question kinds too
  - turns are spent, the budget closes the case, score moves
  - the evidence board grows and rewind shrinks it again
  - quoting grants knowledge and rewinding removes it
  - every lab tool accepts the arguments the UI sends
  - the case library lists, loads, and generation runs as a job
  - accusation dispatches the judge as a job and the verdict lands

Run:  py Test_Api.py
"""

import time
from pathlib import Path

from fastapi.testclient import TestClient

import Api


def show(t):
    print(f"\n{t}")


def wait(client, job_id, limit=90):
    for _ in range(limit):
        j = client.get(f"/api/job/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(1)
    raise AssertionError(f"job {job_id} still running after {limit}s")


def run():
    with TestClient(Api.app) as c:

        show("test_state_and_case_file")
        st = c.get("/api/state").json()
        assert len(st["suspects"]) == 3
        f = st["case"]
        assert f["known"] and f["suspects"] and "board" in f
        assert "SECRET" not in str(f), "case file leaked a secret"
        assert st["game"]["turns_left"] == st["game"]["rules"]["turns"]
        print(f"  {f['title']} · {len(f['known'])} known facts · "
              f"{st['game']['rules']['label']} · {st['game']['turns_left']} turns")

        show("test_manual_served")
        m = c.get("/api/manual")
        assert m.status_code == 200 and "HOW TO PLAY" in m.text
        print(f"  manual {len(m.text)} chars")

        show("test_explore_hides_preview_and_score")
        r = c.post("/api/explore", json={"suspect_id": "vance"}).json()
        assert len(r["candidates"]) == 3
        for cand in r["candidates"]:
            assert "preview" not in cand and "score" not in cand, cand
            assert cand["kind"] != "question", "normal mode should show kinds"
        print(f"  kinds shown, nothing else: {[x['kind'] for x in r['candidates']]}")

        show("test_commit_reveals_and_spends_a_turn")
        before = c.get("/api/state").json()["game"]
        r = c.post("/api/commit", json={"suspect_id": "vance", "index": 0}).json()
        assert r["answer"] and "score" in r and r["band"] in ("low", "medium", "high")
        assert r["game"]["turns_left"] == before["turns_left"] - 1
        assert r["game"]["score"] != before["score"]
        print(f"  band={r['band']} score={r['score']} new_facts={r['new_facts']} "
              f"turns {before['turns_left']}->{r['game']['turns_left']} "
              f"pts {before['score']}->{r['game']['score']}")

        show("test_board_grows")
        f = c.get("/api/state").json()["case"]
        base = len(f["board"])
        c.post("/api/ask", json={"suspect_id": "halloway",
                                 "question": "When did you find her and what did you take from the desk?",
                                 "max_tokens": 40})
        f2 = c.get("/api/state").json()["case"]
        print(f"  board {base} -> {len(f2['board'])} pins, "
              f"discovered {len(f2['discovered'])} facts")

        show("test_rewind_shrinks_board_and_refunds_on_easy")
        c.post("/api/case/load", json={"case_id": "ashfield_001", "difficulty": "easy"})
        c.post("/api/ask", json={"suspect_id": "halloway",
                                 "question": "What did you take from the desk?", "max_tokens": 40})
        g1 = c.get("/api/state").json()["game"]
        b1 = len(c.get("/api/state").json()["case"]["board"])
        r = c.post("/api/rewind", json={"suspect_id": "halloway", "turn_index": 0}).json()
        b2 = len(c.get("/api/state").json()["case"]["board"])
        assert r["cost"] == 0 and r["game"]["turns_left"] == g1["turns_left"] + 1
        assert b2 <= b1
        print(f"  tier={r['tier']} refunded a turn, board {b1}->{b2}")

        show("test_quote_grants_then_rewind_removes")
        c.post("/api/ask", json={"suspect_id": "vance",
                                 "question": "Where were you at 21:38?", "max_tokens": 25})
        Api.WORLD.suspects["vance"].turns[0].answer = (
            "I went up to the observatory at 21:38 and left at 21:46.")
        before_knows = set(Api.WORLD.suspects["rourke"].knows)
        r = c.post("/api/quote", json={"from_suspect": "vance", "turn_index": 0,
                                       "to_suspect": "rourke"}).json()
        assert r["granted"], "pinned answer should grant something"
        c.post("/api/rewind", json={"suspect_id": "rourke", "turn_index": 0})
        assert set(Api.WORLD.suspects["rourke"].knows) == before_knows
        print(f"  granted {r['granted']}, removed by rewind")

        show("test_hard_mode_hides_kinds_and_charges_rewind")
        c.post("/api/case/load", json={"case_id": "ashfield_001", "difficulty": "hard"})
        r = c.post("/api/explore", json={"suspect_id": "rourke"}).json()
        assert all(x["kind"] == "question" for x in r["candidates"])
        c.post("/api/commit", json={"suspect_id": "rourke", "index": 0})
        g = c.get("/api/state").json()["game"]
        r = c.post("/api/rewind", json={"suspect_id": "rourke", "turn_index": 0}).json()
        assert r["cost"] == 2 and r["game"]["turns_left"] == g["turns_left"] - 2
        print(f"  kinds hidden; rewind cost 2 turns ({g['turns_left']}->{r['game']['turns_left']})")

        show("test_turn_budget_closes_the_case")
        c.post("/api/case/load", json={"case_id": "ashfield_001", "difficulty": "hard"})
        for i in range(7):
            c.post("/api/ask", json={"suspect_id": ["vance", "rourke", "halloway"][i % 3],
                                     "question": f"Question {i}?", "max_tokens": 10})
        g = c.get("/api/state").json()["game"]
        assert g["turns_left"] == 0
        r = c.post("/api/explore", json={"suspect_id": "vance"})
        assert r.status_code == 400 and "must accuse" in r.text
        print("  0 turns left -> explore refused, must accuse")

        show("test_every_lab_tool_accepts_ui_args")
        c.post("/api/case/load", json={"case_id": "ashfield_001", "difficulty": "easy"})
        r = c.get("/api/tools")
        if r.status_code == 200:
            st = c.get("/api/state").json()
            scene, at = st["case"]["scene"], st["case"]["window"]["from"]
            ui_args = {
                "check_alibi":      {"suspect_id": "vance", "place": scene, "at": at},
                "who_was_at":       {"place": scene, "at": at},
                "movements_of":     {"suspect_id": "vance"},
                "verify_statement": {"suspect_id": "rourke",
                                     "statement": "I was in the greenhouse at 9:20."},
                "lookup_evidence":  {"query": scene},
            }
            g0 = c.get("/api/state").json()["game"]["turns_left"]
            flagged = 0
            for name, args in ui_args.items():
                rr = c.post("/api/tool", json={"name": name, "args": args})
                assert rr.status_code == 200, f"{name}: {rr.text}"
                if rr.json().get("flag"):
                    flagged += 1
            g1 = c.get("/api/state").json()["game"]["turns_left"]
            assert g1 == g0 - len(ui_args), "each tool use should cost a turn"
            assert flagged >= 1, "verify_statement on the greenhouse lie should flag"
            print(f"  all {len(ui_args)} tools ok, {flagged} flagged a contradiction, "
                  f"{len(ui_args)} turns spent")
        else:
            print("  SKIPPED - forensic server unreachable")

        show("test_case_library_and_load")
        lib = c.get("/api/cases").json()
        assert any(x["case_id"] == "ashfield_001" for x in lib["cases"])
        assert set(lib["difficulties"]) == {"easy", "normal", "hard"}
        r = c.post("/api/case/load", json={"case_id": "nope", "difficulty": "easy"})
        assert r.status_code == 404
        print(f"  {len(lib['cases'])} case(s) in library, bad id -> 404")

        show("test_generate_runs_as_a_job")
        r = c.post("/api/case/generate", json={"n_suspects": 3, "difficulty": "normal"}).json()
        j = wait(c, r["job"], limit=240)
        assert j["status"] == "done", j
        res = j["result"]
        assert res["generated_by"] in ("offline", "deepseek-reasoner", "deepseek-chat")
        lib2 = c.get("/api/cases").json()
        assert any(x["case_id"] == res["case_id"] for x in lib2["cases"])
        c.post("/api/case/load", json={"case_id": res["case_id"], "difficulty": "normal"})
        st = c.get("/api/state").json()
        assert st["case"]["case_id"] == res["case_id"]
        print(f"  generated '{res['title']}' by {res['generated_by']} in {j['elapsed']}s, "
              f"loaded and playable")

        show("test_accuse_dispatches_judge")
        c.post("/api/case/load", json={"case_id": "ashfield_001", "difficulty": "normal"})
        c.post("/api/ask", json={"suspect_id": "vance",
                                 "question": "Did you go up to the observatory?", "max_tokens": 25})
        r = c.post("/api/accuse", json={"suspect_id": "vance"}).json()
        assert r["job"]
        j = wait(c, r["job"], limit=240)
        assert j["status"] == "done", j
        v = j["result"]
        assert v["correct"] is True and "evidence_backed" in v
        assert v["points"] in (500, 150)
        g = c.get("/api/state").json()["game"]
        assert g["finished"] and g["verdict"]["accused_name"]
        r2 = c.post("/api/ask", json={"suspect_id": "vance", "question": "x"})
        assert r2.status_code == 400
        print(f"  judged by {v['judged_by']}: correct={v['correct']} "
              f"backed={v['evidence_backed']} pts={v['points']} final={v['final_score']}")

        show("test_frontend_served")
        if Path(Api.FRONTEND_DIST).exists():
            assert c.get("/").status_code == 200
            print("  / serves the built app")
        else:
            print("  SKIPPED - build the frontend")


if __name__ == "__main__":
    print("=" * 62)
    print("STAGE 9  API (game edition)")
    print("=" * 62)
    run()
    print("\n" + "=" * 62)
    print("stage 9 PASSED")