"""
mock_run.py  -  logic check without a GPU

Builds a fake model and tokenizer that mimic the HuggingFace interface
closely enough to exercise suspects.py and Branching.py end to end.

This cannot verify the real cache maths (that needs the real model), but
it does catch shape bugs, split-point bugs, off-by-ones in crop/rewind,
and anything wrong in the Branching logic.
"""

import json
import sys
import types

import torch

# --- fake torch device so cache_ops' .to("cuda") calls work on CPU -----
DEV = "cpu"

import cache_ops as co
from Extractor import FactExtractor
from Retrieval import HybridRetriever, Timeline, KnowledgeChecker


# ======================================================================
# fake tokenizer
# ======================================================================

class FakeTok:
    """Word-level tokenizer. Deterministic, and never merges across words."""

    def __init__(self):
        self.vocab = {}
        self.inv = {}
        self.eos_token_id = 0
        self._add("<eos>")

    def _add(self, w):
        if w not in self.vocab:
            i = len(self.vocab)
            self.vocab[w] = i
            self.inv[i] = w
        return self.vocab[w]

    def _words(self, text):
        # real tokenizers match <|...|> special tokens greedily as units,
        # so split those out before splitting on whitespace
        import re
        parts = re.split(r"(<\|[a-z_]+\|>)", text)
        words = []
        for p in parts:
            if p.startswith("<|"):
                words.append(p)
            else:
                words.extend(p.replace("\n", " \n ").split(" "))
        return words

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [self._add(w) for w in self._words(text) if w != ""]
        out = types.SimpleNamespace()
        if return_tensors == "pt":
            out.input_ids = torch.tensor([ids])
            out.attention_mask = torch.ones(1, len(ids), dtype=torch.long)
        else:
            out.input_ids = ids
            out.attention_mask = [1] * len(ids)
        return out

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        return " ".join(self.inv.get(int(i), "?") for i in ids)

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, -1)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


# ======================================================================
# fake model
# ======================================================================

class FakeConfig:
    num_hidden_layers = 4
    num_attention_heads = 4
    num_key_value_heads = 2
    hidden_size = 128


class FakeModel:
    """
    Produces a cache with the right shape and grows it correctly.
    Logits are a deterministic function of the cache length so that
    greedy decoding is reproducible.
    """

    def __init__(self, vocab_size=20000):
        self.config = FakeConfig()
        self.vocab_size = vocab_size
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._p

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids=None, past_key_values=None,
                 attention_mask=None, position_ids=None, use_cache=True):
        L = self.config.num_hidden_layers
        H = self.config.num_key_value_heads
        D = self.config.hidden_size // self.config.num_attention_heads
        m = input_ids.shape[1]
        # a real model accepts whatever cache object its version uses, so
        # normalise here exactly as the real code path does
        past = co.normalise(past_key_values)
        n = 0 if past is None else past[0][0].shape[2]

        # sanity: the caller must get mask and positions right
        assert attention_mask.shape[1] == n + m, (
            f"mask is {attention_mask.shape[1]}, expected {n+m}")
        assert position_ids[0, 0].item() == n, (
            f"positions start at {position_ids[0,0].item()}, expected {n}")

        new = []
        for layer in range(L):
            # content depends on the token ids so forks stay distinguishable
            seed = (input_ids.float().mean() + layer) * 0.001
            k_new = torch.full((1, H, m, D), float(seed))
            v_new = torch.full((1, H, m, D), float(seed) + 0.5)
            if past is not None:
                k_old, v_old = past[layer]
                k_new = torch.cat([k_old, k_new], dim=2)
                v_new = torch.cat([v_old, v_new], dim=2)
            new.append((k_new, v_new))

        logits = torch.zeros(1, m, self.vocab_size)
        # deterministic "next token" that eventually emits eos
        pick = (n + m) % 97 + 3
        logits[0, -1, pick] = 10.0

        out = types.SimpleNamespace()
        out.past_key_values = tuple(new)
        out.logits = logits
        return out


# ======================================================================
# run
# ======================================================================

def main():
    import Suspects as sus
    import Branching

    case = json.load(open("case_ashfield.json", encoding="utf-8"))
    tok = FakeTok()
    model = FakeModel()

    ex = FactExtractor(case["entities"])
    tl = Timeline(case["timeline"])
    ret = HybridRetriever(case["entities"], tl)
    kc = KnowledgeChecker(ex, case["public_facts"])

    shared_block = sus.make_shared_block(case)
    privates = [sus.make_private_block(case, sp) for sp in case["suspects"]]
    n_shared, shared_ids = sus.compute_shared_len(tok, shared_block, privates)
    print(f"shared prefix: {n_shared} tokens (computed by LCP)")
    assert n_shared > 0

    shared, _ = co.prefill(
        model, torch.tensor([shared_ids], device=DEV), label="shared")
    assert co.length(shared) == n_shared

    people = {}
    for spec in case["suspects"]:
        s = sus.Suspect(spec, case, shared, shared_ids, model, tok, DEV)
        people[s.id] = s
        print(f"  {s.name:<20} private={s.n_private:>4} base={s.base_len}")

    for s in people.values():
        trimmed = co.crop(s.base_cache, n_shared, label="verify")
        d = co.max_diff(shared, trimmed)
        assert d == 0.0, f"{s.id} diverged by {d}"
    print("shared prefix bit-identical (mock)")

    # ---- ask / rewind ----
    s = people["vance"]
    len0 = co.length(s.cache)
    ids0 = list(s.cached_ids)
    keep = co.fork(s.cache, label="keep")

    s.ask("Where were you at nine?", max_tokens=6)
    assert len(s.turns) == 1
    assert co.length(s.cache) > len0

    s.ask("Did you go upstairs?", max_tokens=6)
    assert len(s.turns) == 2

    s.rewind_last(1)
    assert len(s.turns) == 1, len(s.turns)

    s.rewind_last(1)
    assert len(s.turns) == 0
    assert co.length(s.cache) == len0, (co.length(s.cache), len0)
    assert s.cached_ids == ids0
    assert co.max_diff(keep, s.cache) == 0.0
    print("ask / rewind_last round trip exact")

    # ---- rewind to a middle turn ----
    s.reset()
    for q in ["a?", "b?", "c?"]:
        s.ask(q, max_tokens=4)
    mid = s.turns[1].cache_len_before
    dropped = s.rewind_to(1)
    assert len(dropped) == 2 and len(s.turns) == 1
    assert co.length(s.cache) == mid
    print("rewind_to middle turn correct")

    # ---- snapshot roundtrip ----
    s.reset()
    s.ask("x?", max_tokens=4)
    snap = s.turns[0].snapshot
    assert snap[0][0].device.type == "cpu"
    back = co.restore(snap, device=DEV, label="t")
    assert co.length(back) == s.turns[0].cache_len_before
    print("snapshot roundtrip ok")

    # ---- isolation ----
    a, b = people["rourke"], people["halloway"]
    a.reset(); b.reset()
    a.ask("q?", max_tokens=4)
    b.ask("q?", max_tokens=4)
    b_keep = co.fork(b.cache, label="k")
    b_len = co.length(b.cache)
    a.rewind_last(1)
    assert co.length(b.cache) == b_len
    assert co.max_diff(b_keep, b.cache) == 0.0
    print("suspects isolated from each other's rewinds")

    # ---- Branching ----
    s = people["vance"]
    s.reset()
    s.ask("Where were you at nine?", max_tokens=6)

    before_len = co.length(s.cache)
    before_turns = len(s.turns)
    keepsake = co.fork(s.cache, label="k")

    cands, cost = Branching.explore(s, case, tl, ex, ret, n=3)
    assert len(cands) == 3, len(cands)
    assert cost["forks"] == 3, cost
    assert co.length(s.cache) == before_len, "explore advanced the cache"
    assert len(s.turns) == before_turns, "explore recorded a turn"
    assert co.max_diff(keepsake, s.cache) == 0.0
    print(f"explore: 3 forks, {cost['prefill_tokens']} prefill tokens, "
          f"no trace left")

    for c in cands:
        assert 0.0 <= c.score <= 1.0, c.score
        assert c.band() in ("low", "medium", "high")
    print(f"  scores {[round(c.score, 3) for c in cands]} "
          f"bands {[c.band() for c in cands]}")

    # reproducible
    again, _ = Branching.explore(s, case, tl, ex, ret, n=3)
    for x, y in zip(cands, again):
        assert x.question == y.question
        assert abs(x.score - y.score) < 1e-9
    print("  explore reproducible")

    # commit adopts the previewed fork
    chosen = cands[0]
    ans = Branching.commit(s, chosen, ret, case)
    assert ans == chosen.answer
    assert s.turns[-1].answer == chosen.answer
    assert co.length(s.cache) == co.length(chosen.cache)
    print("  commit adopts the previewed answer exactly")

    # ---- scoring maths ----
    c = Branching.Candidate(
        kind="t", question="q",
        answer="I was in the east hall at 21:50 and called straight away.")
    Branching.score_candidate(c, people["halloway"], case, tl, ex)
    assert c.coverage >= 2, (c.coverage, c.new_entities)
    assert c.verifiable >= 2, c.verifiable

    e = Branching.Candidate(kind="t", question="q",
                            answer="I don't remember anything.")
    Branching.score_candidate(e, people["halloway"], case, tl, ex)
    assert e.coverage == 0 and e.score < c.score
    print(f"  scoring: informative {c.score:.2f} > evasive {e.score:.2f}")

    # ---- knowledge filter ----
    v = kc.violations("I saw the keycard log at 21:38.", people["rourke"].knows)
    assert "ev_keycard" in v, v
    v = kc.violations("I was in the greenhouse.", people["rourke"].knows)
    assert v == set(), v
    print("knowledge filter catches leaks")

    print("\n" + "=" * 60)
    print("MOCK RUN PASSED - logic is sound; real cache maths needs the GPU")
    print(co.COUNTER.summary())


if __name__ == "__main__":
    main()