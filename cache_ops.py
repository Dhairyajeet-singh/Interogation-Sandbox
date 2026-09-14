"""
cache_ops.py
------------
Core KV cache operations for the Branching Interrogation Sandbox.

A cache here is always a plain tuple of (keys, values) pairs, one per layer.
Each tensor is shaped [batch, kv_heads, seq_len, head_dim].
Sequence length lives on dimension 2 - that is the axis we crop and grow.

Design decisions (see stage 0 experiments):
  - crop() CLONES. Slicing returns a view sharing memory with the original,
    which would let one suspect's cache silently corrupt another's.
  - snapshot() takes a device argument, because stage 6 needs to move
    snapshots between GPU and host RAM under a memory budget.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

# A cache is Tuple[Tuple[Tensor, Tensor], ...] but we keep annotations loose
# so this module stays readable.
Cache = tuple


# ======================================================================
# instrumentation
# ======================================================================

@dataclass
class Counter:
    """
    Tracks prefill work and cache events.

    prefill_tokens  - tokens actually pushed through the model
    naive_tokens    - tokens a no-cache implementation would have pushed
    The gap between them is the headline result of this project.
    """
    prefill_tokens: int = 0
    decode_tokens: int = 0
    naive_tokens: int = 0
    events: list = field(default_factory=list)

    def prefill(self, n: int, label: str = "") -> None:
        self.prefill_tokens += n
        self.events.append(("prefill", n, label))

    def decode(self, n: int = 1, label: str = "") -> None:
        self.decode_tokens += n

    def naive(self, n: int, label: str = "") -> None:
        """Record what a from-scratch implementation would have cost."""
        self.naive_tokens += n

    def event(self, kind: str, n: int = 0, label: str = "") -> None:
        """kind: fork | crop | snapshot | restore | evict"""
        self.events.append((kind, n, label))

    @property
    def saved(self) -> int:
        return self.naive_tokens - self.prefill_tokens

    @property
    def saved_pct(self) -> float:
        return 100.0 * self.saved / self.naive_tokens if self.naive_tokens else 0.0

    def summary(self) -> str:
        return (
            f"prefill={self.prefill_tokens} "
            f"naive={self.naive_tokens} "
            f"saved={self.saved} ({self.saved_pct:.0f}%) "
            f"decode={self.decode_tokens}"
        )

    def log(self, kinds: tuple[str, ...] | None = None) -> str:
        rows = [e for e in self.events if kinds is None or e[0] in kinds]
        return "\n".join(f"{k:<9} {n:>6}  {label}" for k, n, label in rows)


COUNTER = Counter()


# ======================================================================
# cache primitives
# ======================================================================

def length(cache: Cache | None) -> int:
    """Number of tokens held in the cache. 0 for an empty cache."""
    return 0 if cache is None else cache[0][0].shape[2]


def n_layers(cache: Cache) -> int:
    return len(cache)


def nbytes(cache: Cache) -> int:
    """Total bytes occupied by this cache."""
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in cache)


def bytes_per_token(cache: Cache) -> float:
    n = length(cache)
    return nbytes(cache) / n if n else 0.0


def normalise(pkv) -> Cache:
    """
    Accept whatever the model returned and hand back a plain tuple.

    transformers <=4.46 returns a tuple of tuples.
    transformers >=4.47 returns a DynamicCache object.
    """
    return pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else pkv


def fork(cache: Cache, label: str = "") -> Cache:
    """
    Independent copy of a cache.

    Used to give several suspects the same prefilled dossier without
    re-reading it. The copy shares no memory with the original, so
    extending one fork cannot affect another.
    """
    out = copy.deepcopy(cache)
    COUNTER.event("fork", length(cache), label)
    return out


def crop(cache: Cache, n: int, label: str = "") -> Cache:
    """
    Return a cache holding only the first n tokens. This is the rewind.

    Clones rather than slicing. A slice would be free but would share
    memory with the source, so later writes through either handle would
    corrupt the other.

    Correctness note: because attention is causal, tokens after position n
    never influenced tokens before it. So a cropped cache is bit-identical
    to a fresh prefill of the first n tokens - not an approximation.
    """
    if n > length(cache):
        raise ValueError(f"cannot crop to {n}, cache holds {length(cache)}")
    out = tuple(
        (k[:, :, :n, :].clone(), v[:, :, :n, :].clone())
        for k, v in cache
    )
    COUNTER.event("crop", n, label)
    return out


def snapshot(cache: Cache, device: str = "cpu", label: str = "") -> Cache:
    """
    Save a cache for later restore.

    device="cpu" moves it off the GPU, freeing VRAM at the cost of a
    transfer when restored. device="cuda" keeps it resident and fast.
    Stage 6's eviction policy chooses between these per snapshot.
    """
    out = tuple(
        (k.detach().to(device).clone(), v.detach().to(device).clone())
        for k, v in cache
    )
    COUNTER.event("snapshot", length(cache), f"{label}->{device}")
    return out


def restore(snap: Cache, device: str = "cuda", label: str = "") -> Cache:
    """Bring a snapshot back onto the compute device."""
    out = tuple((k.to(device), v.to(device)) for k, v in snap)
    COUNTER.event("restore", length(snap), f"{label}->{device}")
    return out


def shares_memory(a: Cache, b: Cache) -> bool:
    """True if the two caches point at the same underlying storage."""
    return a[0][0].data_ptr() == b[0][0].data_ptr()


# ======================================================================
# running the model
# ======================================================================

def prefill(model, ids: torch.Tensor, cache: Cache | None = None,
            label: str = "", count_naive: bool = True):
    """
    Push `ids` through the model, optionally continuing from `cache`.

    Returns (new_cache, logits).

    Two things must be right, and both are silent failures if wrong:
      - attention_mask covers cached tokens PLUS the new ones
      - position_ids continue from where the cache ended, not from 0
        (RoPE rotates each key by its position; wrong positions mean
         the model computes wrong distances between tokens)
    """
    device = next(model.parameters()).device
    n = length(cache)
    m = ids.shape[1]

    with torch.no_grad():
        out = model(
            input_ids=ids,
            past_key_values=cache,
            attention_mask=torch.ones(1, n + m, device=device),
            position_ids=torch.arange(n, n + m, device=device).unsqueeze(0),
            use_cache=True,
        )

    COUNTER.prefill(m, label)
    if count_naive:
        # a cache-blind implementation would have re-read everything
        COUNTER.naive(n + m, label)

    return normalise(out.past_key_values), out.logits


def decode(model, tokenizer, cache: Cache, logits: torch.Tensor,
           max_tokens: int = 40, stop_ids: tuple[int, ...] | None = None):
    """
    Greedy-generate from the current cache, one token at a time.

    Each step feeds exactly ONE token, because the cache already holds
    everything before it. Returns (text, new_cache).
    """
    device = next(model.parameters()).device
    if stop_ids is None:
        stop_ids = tuple(
            i for i in (tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|im_end|>"))
            if i is not None
        )

    nxt = logits[:, -1, :].argmax(-1, keepdim=True)
    produced: list[int] = []

    for _ in range(max_tokens):
        if nxt.item() in stop_ids:
            break
        produced.append(nxt.item())

        n = length(cache)
        with torch.no_grad():
            out = model(
                input_ids=nxt,
                past_key_values=cache,
                attention_mask=torch.ones(1, n + 1, device=device),
                position_ids=torch.tensor([[n]], device=device),
                use_cache=True,
            )
        cache = normalise(out.past_key_values)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)

    COUNTER.decode(len(produced))
    return tokenizer.decode(produced), cache


# ======================================================================
# tokenisation
# ======================================================================

def split_ids(tokenizer, system_text: str, user_text: str, device: str = "cuda"):
    """
    Build chat-formatted token ids and return (full_ids, n_shared).

    Tokenises the WHOLE string once, then reports where the shared system
    block ends, so callers can split by index.

    Never tokenise two strings separately and concatenate them: a trailing
    space on one merges with the leading token of the next, the ids no
    longer line up, and cached prefixes silently stop matching.
    """
    shared_only = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}],
        tokenize=False,
    )
    full = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text},
         {"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_ids = tokenizer(
        full, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    n_shared = len(tokenizer(shared_only, add_special_tokens=False).input_ids)
    return full_ids, n_shared