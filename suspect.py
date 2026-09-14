"""
stage1_suspects.py  (fixed)
---------------------------
Stage 1: three suspects forked from ONE prefilled dossier.

    [ dossier ][ secret ][ question ]
      shared     private    private
      ^^^^^^^^
      prefilled once, forked N times

Fixes over the first version:
  1. Honest naive accounting. The old version let every per-question fork
     add (n + m) to naive, which counted re-reading the whole conversation
     before every single question. No sane implementation would do that,
     so the 85% was inflated. Naive is now computed explicitly at the end:
     each suspect reads dossier + secret once, then each question once.
  2. Verifies the shared prefix is bit-identical across all suspects,
     so a bad split point fails loudly instead of silently diverging.
  3. split_three finds its boundaries with a sentinel instead of guessing
     where the chat template ends, and asserts they are sane.
  4. ask() no longer forks per question - the turn belongs to the suspect.
     (Speculative forking arrives in stage 5, where it is the point.)

Run:  py stage1_suspects.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import cache_ops as co

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda"
SENTINEL = "\u0001SPLIT\u0001"      # bytes no tokenizer will merge across

print("loading model...")
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(
    NAME, torch_dtype=torch.float16, attn_implementation="eager"
).to(DEV)
model.eval()


# ======================================================================
# three-part tokenisation
# ======================================================================

def ids_of(text):
    return tok(text, add_special_tokens=False).input_ids


def split_three(dossier, secret, question, device=DEV):
    """
    Tokenise the whole prompt ONCE and return (ids, n1, n2) where

        ids[:, :n1]     dossier   - identical for every suspect
        ids[:, n1:n2]   secret    - private, prefilled per suspect
        ids[:, n2:]     question  - private, prefilled per turn

    The boundary is found by inserting a sentinel, tokenising, and
    locating the sentinel's token ids. Exact - no guessing where the
    chat template ends, no string concatenation.
    """
    marked = tok.apply_chat_template(
        [{"role": "system", "content": f"{dossier}{SENTINEL}{secret}"},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    clean = marked.replace(SENTINEL, "")

    sent_ids = ids_of(SENTINEL)
    marked_ids = ids_of(marked)

    n1 = None
    for i in range(len(marked_ids) - len(sent_ids) + 1):
        if marked_ids[i:i + len(sent_ids)] == sent_ids:
            n1 = i
            break
    if n1 is None:
        raise RuntimeError("sentinel not found - tokenizer merged across it")

    system_only = tok.apply_chat_template(
        [{"role": "system", "content": f"{dossier}{SENTINEL}{secret}"}],
        tokenize=False,
    )
    n2 = len(ids_of(system_only.replace(SENTINEL, "")))

    ids = tok(clean, return_tensors="pt",
              add_special_tokens=False).input_ids.to(device)

    assert 0 < n1 < n2 < ids.shape[1], f"bad split: {n1} {n2} {ids.shape[1]}"
    return ids, n1, n2


# ======================================================================
# suspect
# ======================================================================

class Suspect:
    def __init__(self, spec, shared_cache, dossier, n_dossier):
        self.id = spec["id"]
        self.name = spec["name"]
        self.role = spec["role"]
        self.guilty = spec["guilty"]
        self.secret = spec["secret"]
        self.dossier = dossier
        self.n_dossier = n_dossier
        self.transcript = []

        ids, n1, n2 = split_three(dossier, self.secret, "placeholder")
        assert n1 == n_dossier, (
            f"{self.id}: dossier tokenised to {n1}, expected {n_dossier}. "
            "The shared prefix must be identical for every suspect."
        )
        self.n_secret = n2 - n1

        # fork the shared dossier, prefill only THIS suspect's secret
        self.base, _ = co.prefill(
            model, ids[:, n1:n2], co.fork(shared_cache, label=self.id),
            label=f"{self.id}:secret", count_naive=False,
        )
        self.cache = self.base

    def ask(self, question, max_tokens=45):
        """Ask a question. The turn is appended to this suspect's cache."""
        ids, _, n2 = split_three(self.dossier, self.secret, question)
        q_ids = ids[:, n2:]

        cache, logits = co.prefill(
            model, q_ids, self.cache,
            label=f"{self.id}:q", count_naive=False,
        )
        answer, cache = co.decode(model, tok, cache, logits, max_tokens)
        self.cache = cache
        self.transcript.append((question, answer.strip()))
        return answer.strip()

    def reset(self):
        """Back to dossier + secret. Stage 2 replaces this with snapshots."""
        self.cache = co.crop(self.base, co.length(self.base),
                             label=f"{self.id}:reset")
        self.transcript = []
        return self


# ======================================================================
# checks
# ======================================================================

def verify_shared_prefix(shared, suspects):
    """
    Every suspect's cache must begin with a bit-identical copy of the
    shared dossier, across all layers. A wrong split point shows up here.
    """
    n = co.length(shared)
    for s in suspects:
        d = max(
            (shared[L][0][:, :, :n, :].float()
             - s.base[L][0][:, :, :n, :].float()).abs().max().item()
            for L in range(co.n_layers(shared))
        )
        assert d == 0.0, f"{s.id}: shared prefix diverged by {d:.2e}"
        assert not co.shares_memory(shared, s.base), \
            f"{s.id}: shares memory with the shared cache"
    print(f"  shared prefix verified: {n} tokens, "
          f"all layers, diff 0.00e+00, no shared storage")


# ======================================================================
# main
# ======================================================================

def main():
    case = json.load(open("case_ashfield.json", encoding="utf-8"))
    dossier = case["dossier"]
    specs = case["suspects"]
    questions = case["probe_questions"]

    # ---- prefill the dossier exactly ONCE ----
    probe_ids, n_dossier, _ = split_three(dossier, "x", "x")
    shared, _ = co.prefill(model, probe_ids[:, :n_dossier],
                           label="dossier", count_naive=False)

    print(f"\ndossier prefilled once: {co.length(shared)} tokens "
          f"({co.nbytes(shared)/1024/1024:.2f} MB, "
          f"{co.bytes_per_token(shared):.0f} B/token)")
    print(f"tail of shared block: "
          f"{tok.decode(probe_ids[0, n_dossier-10:n_dossier])!r}\n")

    # ---- fork one suspect per spec ----
    suspects = [Suspect(s, shared, dossier, n_dossier) for s in specs]
    for s in suspects:
        print(f"  {s.name:<20} secret {s.n_secret:>3} tok  "
              f"base len {co.length(s.base)}")
    verify_shared_prefix(shared, suspects)

    # ---- interrogate ----
    for q in questions:
        print(f"\n{'='*66}\nQ: {q}\n{'='*66}")
        for s in suspects:
            print(f"{s.name:<20} {s.ask(q)}")

    # ---- honest savings ----
    real = co.COUNTER.prefill_tokens

    naive = 0
    for s in suspects:
        naive += n_dossier + s.n_secret          # each reads the dossier itself
        for q, _ in s.transcript:
            ids, _, n2 = split_three(dossier, s.secret, q)
            naive += ids.shape[1] - n2           # plus each question

    print(f"\n{'='*66}")
    print("PREFILL ACCOUNTING")
    print(f"  forked : {real:>6} tokens")
    print(f"  naive  : {naive:>6} tokens")
    print(f"  saved  : {naive - real:>6} tokens "
          f"({100*(naive-real)/naive:.1f}%)")
    print(f"\n  the whole saving is the dossier read once, not "
          f"{len(suspects)} times:")
    print(f"    {n_dossier} x {len(suspects)-1} = "
          f"{n_dossier*(len(suspects)-1)} tokens")
    print(f"\n  decode tokens (forking does not affect these): "
          f"{co.COUNTER.decode_tokens}")

    print("\ncache events:")
    print(co.COUNTER.log(kinds=("fork", "crop")))


if __name__ == "__main__":
    main()