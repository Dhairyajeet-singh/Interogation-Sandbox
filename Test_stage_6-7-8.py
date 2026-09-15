"""
Test_Stages678.py
-----------------
Acceptance tests for:

  Stage 6  snapshot budget, three tiers, eviction, and correct restore
           from every tier
  Stage 7  composure driving temperature, cross-suspect quoting granting
           knowledge, and rewind taking that knowledge away again
  Stage 8  the MCP forensic server over both transports

Run:  py Test_Stages678.py
"""

import json

import torch

import cache_ops as co
import Interrogation
from Extractor import FactExtractor
from Model_setup import load_model, tolerance
from Retrieval import HybridRetriever, Timeline
from Snapshots import (SnapshotStore, quantise, dequantise, qbytes,
                       GPU, INT8, CPU, DROPPED)
from Suspects import (Suspect, compute_shared_len, make_shared_block,
                      make_private_block)

tok, model, DEVICE = load_model()
TOL = tolerance(model)

case = json.load(open("Case_Ashfield.json", encoding="utf-8"))
ex = FactExtractor(case["entities"])
tl = Timeline(case["timeline"])
ret = HybridRetriever(case["entities"], tl)

shared_block = make_shared_block(case)
privates = [make_private_block(case, sp) for sp in case["suspects"]]
n_shared, shared_ids = compute_shared_len(tok, shared_block, privates)
shared, _ = co.prefill(model, torch.tensor([shared_ids], device=DEVICE),
                       label="shared")


def rel_error(a, b):
    """
    Difference scaled by the size of the numbers involved.

    Quantisation error is proportional to magnitude, so an absolute
    threshold is the wrong tool: 0.87 is tiny next to values reaching 220
    and enormous next to values reaching 1.
    """
    scale = max(k.abs().amax().item() for k, v in a)
    return co.max_diff(a, b) / max(scale, 1e-6)


def fresh_suspects(store=None):
    return {
        spec["id"]: Suspect(spec, case, shared, shared_ids, model, tok,
                            DEVICE, store=store)
        for spec in case["suspects"]
    }


print(f"shared block: {n_shared} tokens, tol={TOL}\n")


# ======================================================================
# STAGE 6
# ======================================================================

def test_int8_roundtrip_error_is_small():
    """
    Quantising to int8 is lossy. Measure how lossy, rather than assuming.
    The number belongs in the report.
    """
    cache, _ = co.prefill(model, torch.tensor([shared_ids], device=DEVICE))

    q = quantise(cache)
    back = dequantise(q, cache[0][0].dtype, DEVICE)

    err = co.max_diff(cache, back)
    biggest = max(k.abs().amax().item() for k, v in cache)
    ratio = qbytes(q) / co.nbytes(cache)

    print(f"  int8 max error {err:.4f} against max |value| {biggest:.2f} "
          f"({100*err/biggest:.2f}%)")
    print(f"  size {ratio:.0%} of fp16")

    # int8 against fp16 is ~50%; against fp32 it is ~25%
    assert ratio < 0.60, f"int8 did not shrink the cache: {ratio:.0%}"
    assert err < biggest * 0.02, "int8 error above 2% of the value range"


def test_store_serves_from_every_tier():
    """
    A snapshot must come back correctly from every tier.

    Note which tiers are lossless and which are not:
      gpu   exact
      cpu   exact IF it came straight from gpu
      int8  lossy, and anything demoted THROUGH int8 inherits that loss -
            you cannot recover precision you already threw away
    """
    cache, _ = co.prefill(model, torch.tensor([shared_ids], device=DEVICE))

    # --- gpu: exact
    store = SnapshotStore(device=DEVICE)
    store.put("k", cache)
    assert store.entries["k"].tier == GPU
    assert co.max_diff(store.get("k"), cache) == 0.0

    # --- int8: lossy, but small relative to the value range
    store._demote(store.entries["k"])
    assert store.entries["k"].tier == INT8
    e_int8 = rel_error(cache, store.get("k"))
    assert e_int8 < 0.02, f"int8 relative error {e_int8:.4f}"

    # --- cpu reached through int8: inherits the int8 error
    store._demote(store.entries["k"])
    assert store.entries["k"].tier == CPU
    e_via = rel_error(cache, store.get("k"))
    assert abs(e_via - e_int8) < 1e-6, "cpu should preserve what int8 left"

    # --- cpu reached directly from gpu: lossless
    direct = SnapshotStore(device=DEVICE, use_int8=False)
    direct.put("k", cache)
    direct._demote(direct.entries["k"])
    assert direct.entries["k"].tier == CPU
    assert co.max_diff(direct.get("k"), cache) == 0.0

    # --- dropped: gone
    store._demote(store.entries["k"])
    assert store.entries["k"].tier == DROPPED
    assert store.get("k") is None

    print(f"  gpu exact | int8 rel err {e_int8:.4%} | "
          f"cpu-direct exact | dropped -> None")
    print(f"  {store.stats.summary()}")


def test_budget_forces_demotion():
    """
    Put more snapshots in than the budget allows and watch the store push
    the least recently used ones down the tiers.
    """
    cache, _ = co.prefill(model, torch.tensor([shared_ids], device=DEVICE))
    one = co.nbytes(cache)

    store = SnapshotStore(gpu_budget_bytes=int(one * 2.2), device=DEVICE)
    for i in range(6):
        store.put(f"s{i}", cache)

    counts = store.tier_counts()
    print(f"  budget {one*2.2/1e6:.1f}MB, 6 snapshots of {one/1e6:.1f}MB each")
    print(f"  -> {store.report()}")

    assert store.gpu_bytes() <= store.budget, "still over budget after eviction"
    assert counts[GPU] + counts[INT8] < 6, "nothing was demoted"
    assert counts[CPU] + counts[DROPPED] > 0, "nothing left the GPU"


def test_eviction_is_least_recently_used():
    """The snapshot nobody has touched should be the one that moves."""
    cache, _ = co.prefill(model, torch.tensor([shared_ids], device=DEVICE))
    one = co.nbytes(cache)
    store = SnapshotStore(gpu_budget_bytes=int(one * 2.2), device=DEVICE,
                          use_int8=False)

    store.put("old", cache)
    store.put("mid", cache)
    store.get("old")               # touch it, so "mid" becomes coldest
    store.put("new", cache)

    assert store.entries["old"].tier == GPU, store.report()
    assert store.entries["mid"].tier != GPU, "LRU picked the wrong victim"
    print(f"  touched entry kept, untouched one demoted ({store.report()})")


def test_suspect_rewind_survives_eviction():
    """
    The behavioural gate for stage 6. Force every snapshot out of the
    GPU, then rewind. The suspect must land in exactly the right state
    whichever tier the snapshot came back from.
    """
    tiny = SnapshotStore(gpu_budget_bytes=1, device=DEVICE)   # nothing fits
    people = fresh_suspects(store=tiny)
    s = people["vance"]

    s.ask("Where were you at nine?", max_tokens=20)
    target_len = s.turns[0].cache_len_before
    keep = co.fork(s.cache, label="after-turn")

    s.ask("Did you go upstairs?", max_tokens=20)
    assert len(s.turns) == 2

    tier = s.restore_snapshot(0)
    assert len(s.turns) == 0
    assert co.length(s.cache) == target_len, (co.length(s.cache), target_len)

    # and the suspect still works from the restored state
    answer, _, _ = s.ask("Anything else?", max_tokens=20)
    assert isinstance(answer, str) and answer
    print(f"  restored from tier '{tier}' at len {target_len}, still answers")
    print(f"  {tiny.stats.summary()}")


# ======================================================================
# STAGE 7
# ======================================================================

def test_temperature_tracks_composure():
    assert Interrogation.temperature_for(1.0) == 0.0
    assert Interrogation.temperature_for(0.0) == Interrogation.MAX_TEMPERATURE
    assert (Interrogation.temperature_for(0.2)
            > Interrogation.temperature_for(0.8))
    print("  calm -> greedy, cornered -> sampled")


def test_composure_falls_on_contradiction_and_recovers():
    people = fresh_suspects()
    s = people["rourke"]
    sess = Interrogation.Session()

    start = s.composure
    sess.check_answer(s, case, tl, ex, "q",
                      "I was in the greenhouse at 9:20.", 0)
    after_hit = s.composure

    sess.check_answer(s, case, tl, ex, "q", "I have nothing to add.", 1)
    after_calm = s.composure

    assert after_hit < start, (start, after_hit)
    assert after_calm > after_hit
    assert len(sess.hits_for("rourke")) == 1
    print(f"  {start:.2f} -> {after_hit:.2f} (caught) -> {after_calm:.2f} (eased)")


def test_quoting_grants_knowledge_and_rewind_removes_it():
    """
    The sharpest test in the project. Telling A what B said gives A new
    knowledge; rewinding past that turn takes it back. A prompt-level
    "forget that" could not do this.
    """
    people = fresh_suspects()
    sess = Interrogation.Session()
    vance, rourke = people["vance"], people["rourke"]

    # give Vance a concrete statement to be quoted. The small model may
    # or may not name a time on its own, so the answer is pinned here -
    # what is under test is the grant/rewind mechanism, not the wording.
    vance.ask("Where were you at 21:38?", max_tokens=20)
    vance.turns[-1].answer = "I went up to the observatory at 21:38."

    before = list(rourke.knows)
    question, granted = sess.quote(vance, 0, rourke, case, ex)
    assert granted, "quoting should surface at least one new entity"
    assert "observatory" in question.lower() or "21:38" in question

    answer, note = Interrogation.ask_with_composure(
        rourke, sess, case, tl, ex, ret, question, granted=granted,
        max_tokens=25)

    for eid in granted:
        assert eid in rourke.knows, f"{eid} was not granted"
    assert rourke.turns[-1].granted == granted

    rourke.rewind_last(1)
    for eid in granted:
        assert eid not in rourke.knows, f"{eid} survived the rewind"
    assert rourke.knows == before, (rourke.knows, before)
    print(f"  granted {granted} on quote, all removed by rewind")


def test_rewind_restores_composure():
    people = fresh_suspects()
    sess = Interrogation.Session()
    s = people["halloway"]

    s.composure = 0.9
    Interrogation.ask_with_composure(
        s, sess, case, tl, ex, ret, "Where were you at 21:50?", max_tokens=20)
    s.composure = 0.2                       # simulate being rattled

    s.rewind_last(1)
    assert abs(s.composure - 0.9) < 1e-9, s.composure
    print("  composure restored with the cache")


# ======================================================================
# STAGE 8
# ======================================================================

def test_mcp_stdio_discovery_and_calls():
    from Mcp_client import ForensicsClient, Forensics

    client = ForensicsClient()                     # spawns the server
    tools = dict(client.list_tools())
    assert tools, "server offered no tools"
    assert any("check_alibi" in n for n in tools)

    f = Forensics(client)
    good = f.check_alibi("vance", "observatory", "21:40")
    bad = f.check_alibi("rourke", "greenhouse", "21:30")
    assert good["verdict"] == "supported", good
    assert bad["verdict"] == "contradicted", bad

    ver = f.verify_statement("rourke", "I was in the greenhouse at 9:20.")
    assert ver["contradicted"] is True, ver

    text = client.read_resource("case://timeline")
    assert "per_vance" in text
    print(f"  discovered {len(tools)} tools over stdio, calls and resource ok")


def test_mcp_tools_are_discovered_not_hardcoded():
    """
    The agent resolves tool names from the server's advertised list. Add a
    tool to Mcp_server.py and it becomes usable with no change here.
    """
    from Mcp_client import ForensicsClient, Forensics
    f = Forensics(ForensicsClient())
    names = f.available()
    assert f._resolve("check_alibi") in names
    try:
        f._resolve("does_not_exist")
        raise AssertionError("should have refused an unknown tool")
    except KeyError:
        pass
    print(f"  resolves against the live list: {sorted(names)[:3]} ...")


# ======================================================================

if __name__ == "__main__":
    groups = [
        ("STAGE 6  snapshot budget and eviction", [
            test_int8_roundtrip_error_is_small,
            test_store_serves_from_every_tier,
            test_budget_forces_demotion,
            test_eviction_is_least_recently_used,
            test_suspect_rewind_survives_eviction,
        ]),
        ("STAGE 7  composure, quoting, contradiction", [
            test_temperature_tracks_composure,
            test_composure_falls_on_contradiction_and_recovers,
            test_quoting_grants_knowledge_and_rewind_removes_it,
            test_rewind_restores_composure,
        ]),
        ("STAGE 8  MCP forensic tools", [
            test_mcp_stdio_discovery_and_calls,
            test_mcp_tools_are_discovered_not_hardcoded,
        ]),
    ]

    for title, tests in groups:
        print("=" * 62)
        print(title)
        print("=" * 62)
        for t in tests:
            print(f"\n{t.__name__}")
            t()
        print()

    print("=" * 62)
    print("stages 6, 7 and 8 PASSED")
    print(co.COUNTER.summary())