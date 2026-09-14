"""
suspects.py   -  Stages 2 and 3   (LCP rewrite)
-----------------------------------------------
The Suspect class. Holds one KV cache, the exact token ids that cache was
built from, and a snapshot per turn so any turn can be rewound to.

WHY THIS WAS REWRITTEN
----------------------
The first version tokenised the whole prompt and sliced it at fixed
indices, assuming the shared block always ended on a clean token
boundary. It does not. A BPE tokenizer can merge the last character of
one block with the first character of the next, so the index that looks
like "end of the shared block" can land inside a token. On Qwen's real
tokenizer that is exactly what happened.

The fix is to stop guessing. We compare token id sequences directly and
reuse however many leading tokens genuinely match. This is
longest-common-prefix reuse, which is what vLLM and SGLang do
internally. It is correct for any tokenizer, template or prompt.

    cached:  [a b c d e f g]      what the cache already holds
    wanted:  [a b c d X Y]        what this turn needs
             ^^^^^^^              4 tokens in common
    -> crop the cache to 4, prefill [X Y]

Forking, rewinding, and sharing the case file across suspects are all
special cases of that one rule.
"""

import torch

import cache_ops as co


# ======================================================================
# token helpers
# ======================================================================

def common_prefix_len(a, b):
    """How many leading token ids two sequences share."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build_base_ids(tok, shared_block, private_block):
    """
    The system block only, with no question yet.

    A suspect's base cache is built from this, and every later turn
    extends it. Rendered through the real chat template so the model
    sees the format it was trained on.
    """
    text = tok.apply_chat_template(
        [{"role": "system", "content": f"{shared_block}\n\n{private_block}"}],
        tokenize=False,
    )
    return tok(text, add_special_tokens=False).input_ids


# ChatML markers. Qwen, Llama-3 chat and most instruct models use these.
# If you swap to a model with a different template, change them here.
USER_OPEN = "<|im_start|>user\n"
TURN_CLOSE = "<|im_end|>\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n"


def turn_suffix_ids(tok, turn_block, first_turn):
    """
    The token ids that extend a conversation by one user turn.

    Built as ids, not by re-rendering the whole template. Re-rendering
    would mean re-tokenising the model's own previous answers, and a
    decode/encode round trip does not reliably give back the same tokens.
    Appending ids directly means the cached prefix always matches exactly.
    """
    head = "" if first_turn else TURN_CLOSE      # close the assistant turn
    text = f"{head}{USER_OPEN}{turn_block}{TURN_CLOSE}{ASSISTANT_OPEN}"
    return tok(text, add_special_tokens=False).input_ids


# ======================================================================
# prompt blocks
# ======================================================================

def make_shared_block(case):
    """The part every suspect sees: public brief plus public facts."""
    lines = [case["public_brief"], "", "Known to everyone:"]
    for eid in case["public_facts"]:
        lines.append("- " + case["entities"][eid]["text"])
    return "\n".join(lines)


def make_private_block(case, spec):
    """Persona, secret, and the facts only this suspect knows."""
    lines = [spec["persona"], "", spec["secret"], "", "Known only to you:"]
    for eid in spec["knows"]:
        lines.append("- " + case["entities"][eid]["text"])
    return "\n".join(lines)


def make_turn_block(question, retrieved_lines=None):
    """One turn's user message: retrieved facts, then the question."""
    if not retrieved_lines:
        return f"Detective: {question}"
    facts = "\n".join("- " + line for line in retrieved_lines)
    return (f"Relevant to this question, from what you know:\n{facts}\n\n"
            f"Detective: {question}")


def compute_shared_len(tok, shared_block, private_blocks):
    """
    How many leading tokens every suspect has in common.

    Computed, not assumed: tokenise each suspect's base prompt and take
    the longest prefix they all agree on. Those tokens get prefilled once
    and forked, which is the saving this project is built on.
    """
    seqs = [build_base_ids(tok, shared_block, pb) for pb in private_blocks]
    n = len(seqs[0])
    for other in seqs[1:]:
        n = min(n, common_prefix_len(seqs[0], other))
    return n, seqs[0][:n]


# ======================================================================
# turn record
# ======================================================================

class Turn:
    """One question and answer, plus the cache state that preceded it."""

    def __init__(self, index, question, answer, ids_before,
                 retrieved, snapshot):
        self.index = index
        self.question = question
        self.answer = answer
        self.ids_before = list(ids_before)
        self.retrieved = list(retrieved or [])
        self.snapshot = snapshot        # cache as it was BEFORE this turn

    @property
    def cache_len_before(self):
        return len(self.ids_before)

    def __repr__(self):
        return f"<Turn {self.index}: {self.question[:32]!r}>"


# ======================================================================
# suspect
# ======================================================================

class Suspect:
    def __init__(self, spec, case, shared_cache, shared_ids,
                 model, tok, device="cuda"):
        """
        shared_cache / shared_ids: the prefilled common prefix and the
        token ids it was built from. Every suspect forks this.
        """
        self.spec = spec
        self.id = spec["id"]
        self.name = spec["name"]
        self.role = spec["role"]
        self.guilty = spec["guilty"]
        self.knows = list(spec["knows"])
        self.lies = spec.get("lies", [])

        self.case = case
        self.model = model
        self.tok = tok
        self.device = device

        self.shared_block = make_shared_block(case)
        self.private_block = make_private_block(case, spec)

        base_ids = build_base_ids(tok, self.shared_block, self.private_block)
        n_common = common_prefix_len(shared_ids, base_ids)
        assert n_common == len(shared_ids), (
            f"{self.id}: only {n_common} of {len(shared_ids)} shared tokens "
            "match. The shared block must be identical for every suspect."
        )

        # fork the shared prefix, prefill only what this suspect adds
        cache = co.fork(shared_cache, label=self.id)
        extra = base_ids[n_common:]
        if extra:
            cache, _ = co.prefill(
                model, torch.tensor([extra], device=device), cache,
                label=f"{self.id}:private",
            )

        self.base_cache = cache
        self.base_ids = list(base_ids)
        self.cache = cache
        self.cached_ids = list(base_ids)
        self.n_private = len(extra)
        self.turns = []

        # 1.0 calm, 0.0 cornered. Stage 7 ties this to sampling temperature.
        self.composure = 0.85

    # ------------------------------------------------------------------
    @property
    def base_len(self):
        return len(self.base_ids)

    def transcript_text(self):
        return "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in self.turns)

    # ------------------------------------------------------------------
    def _reuse(self, cache, cached_ids, wanted_ids, label):
        """
        Reuse as much of `cache` as genuinely matches `wanted_ids`.

        Crops down to the common prefix, then prefills the rest.
        Returns (cache, logits, n_reused, n_prefilled).
        """
        n_common = common_prefix_len(cached_ids, wanted_ids)

        if n_common < len(cached_ids):
            cache = co.crop(cache, n_common, label=f"{label}:trim")

        new_ids = wanted_ids[n_common:]
        if not new_ids:
            raise ValueError("nothing new to prefill; the prompt is unchanged")

        cache, logits = co.prefill(
            self.model, torch.tensor([new_ids], device=self.device),
            cache, label=label,
        )
        return cache, logits, n_common, len(new_ids)

    # ------------------------------------------------------------------
    def ask(self, question, retrieved_lines=None, max_tokens=50,
            temperature=0.0, seed=None, cache=None, cached_ids=None,
            record=True):
        """
        Ask a question and get an answer.

        cache / cached_ids: run against this state instead of the live one.
                            Stage 5 passes a fork here to try a question
                            without committing to it.
        record:             False means do not advance the live cache and
                            do not append to the transcript.

        Returns (answer, new_cache, new_ids).
        """
        working = self.cache if cache is None else cache
        working_ids = self.cached_ids if cached_ids is None else cached_ids

        snap = None
        if record:
            snap = co.snapshot(working, device="cpu",
                               label=f"{self.id}:t{len(self.turns)}")

        turn_block = make_turn_block(question, retrieved_lines)
        first = len(working_ids) == len(self.base_ids)
        suffix = turn_suffix_ids(self.tok, turn_block, first_turn=first)
        wanted = list(working_ids) + suffix

        new_cache, logits, n_reused, n_new = self._reuse(
            working, working_ids, wanted, label=f"{self.id}:turn"
        )
        assert n_reused == len(working_ids), (
            f"expected to reuse all {len(working_ids)} cached tokens, "
            f"reused {n_reused}"
        )

        answer, new_cache, produced = co.decode(
            self.model, self.tok, new_cache, logits,
            max_tokens=max_tokens, temperature=temperature, seed=seed,
        )
        answer = answer.strip()

        # the cache holds the prompt plus exactly these generated ids
        new_ids = list(wanted) + list(produced)
        assert len(new_ids) == co.length(new_cache), (
            f"id tracking drifted: {len(new_ids)} ids vs "
            f"{co.length(new_cache)} cached tokens"
        )

        if record:
            self.turns.append(Turn(
                index=len(self.turns),
                question=question,
                answer=answer,
                ids_before=working_ids,
                retrieved=retrieved_lines,
                snapshot=snap,
            ))
            self.cache = new_cache
            self.cached_ids = new_ids

        return answer, new_cache, new_ids

    # ------------------------------------------------------------------
    # stage 2: rewind
    # ------------------------------------------------------------------
    def rewind_to(self, turn_index):
        """
        Forget everything from turn `turn_index` onward.

        The cache is cropped back to the length it had before that turn,
        so the erased tokens stop existing. The model cannot be influenced
        by them because they are no longer in its memory.
        """
        if not (0 <= turn_index < len(self.turns)):
            raise IndexError(
                f"turn {turn_index} out of range (have {len(self.turns)})"
            )

        target = self.turns[turn_index]
        self.cache = co.crop(self.cache, target.cache_len_before,
                             label=f"{self.id}:rewind->t{turn_index}")
        self.cached_ids = list(target.ids_before)

        dropped = self.turns[turn_index:]
        self.turns = self.turns[:turn_index]
        return dropped

    def rewind_last(self, n=1):
        """Drop the last n turns."""
        if n > len(self.turns):
            raise IndexError(f"cannot drop {n} of {len(self.turns)} turns")
        return self.rewind_to(len(self.turns) - n)

    def reset(self):
        """Back to the base state: shared block plus this suspect's block."""
        if co.length(self.cache) > self.base_len:
            self.cache = co.crop(self.cache, self.base_len,
                                 label=f"{self.id}:reset")
        self.cached_ids = list(self.base_ids)
        self.turns = []

    def restore_snapshot(self, turn_index, to_device=None):
        """
        Reload a saved snapshot instead of cropping.

        Same result as rewind_to, but used by stage 6 when the live cache
        has been evicted and must come back from host RAM.
        """
        target = self.turns[turn_index]
        self.cache = co.restore(target.snapshot,
                                device=to_device or self.device,
                                label=f"{self.id}:restore t{turn_index}")
        self.cached_ids = list(target.ids_before)
        self.turns = self.turns[:turn_index]