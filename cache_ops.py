"""
cache_ops.py
------------
Core KV cache operations for the Branching Interrogation Sandbox.

A cache is always a plain tuple of (keys, values) pairs, one per layer.
Each tensor is shaped [batch, kv_heads, seq_len, head_dim].
Sequence length is dimension 2 - that is the axis we crop and grow.

Design decisions, all established by the stage 0 experiments:
  - crop() CLONES. A raw slice shares memory with its source, so writing
    through either handle corrupts the other.
  - snapshot() takes a device, because stage 6 moves snapshots between
    GPU and host RAM under a memory budget.
  - decode() supports temperature, because stage 7 ties a suspect's
    composure to how erratic their speech becomes. temperature=0 is
    greedy and fully deterministic, which stage 5 needs for stable scores.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

Cache = tuple


# ======================================================================
# instrumentation
# ======================================================================

@dataclass
class Counter:
    """
    Tracks prefill work and cache events.

    prefill_tokens  - tokens actually pushed through the model
    naive_tokens    - what a cache-blind implementation would have pushed
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
        self.naive_tokens += n

    def event(self, kind: str, n: int = 0, label: str = "") -> None:
        """kind: fork | crop | snapshot | restore | evict"""
        self.events.append((kind, n, label))

    def count(self, kind: str) -> int:
        return sum(1 for k, _, _ in self.events if k == kind)

    @property
    def saved(self) -> int:
        return self.naive_tokens - self.prefill_tokens

    @property
    def saved_pct(self) -> float:
        return 100.0 * self.saved / self.naive_tokens if self.naive_tokens else 0.0

    def summary(self) -> str:
        return (f"prefill={self.prefill_tokens} decode={self.decode_tokens} "
                f"forks={self.count('fork')} crops={self.count('crop')}")

    def log(self, kinds: tuple | None = None, last: int | None = None) -> str:
        rows = [e for e in self.events if kinds is None or e[0] in kinds]
        if last:
            rows = rows[-last:]
        return "\n".join(f"{k:<9} {n:>6}  {label}" for k, n, label in rows)

    def reset(self) -> None:
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.naive_tokens = 0
        self.events = []


COUNTER = Counter()


# ======================================================================
# inspecting a cache
# ======================================================================

def length(cache: Cache | None) -> int:
    """How many tokens this cache holds. 0 if empty."""
    return 0 if cache is None else cache[0][0].shape[2]


def n_layers(cache: Cache) -> int:
    return len(cache)


def nbytes(cache: Cache) -> int:
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in cache)


def bytes_per_token(cache: Cache) -> float:
    n = length(cache)
    return nbytes(cache) / n if n else 0.0


def normalise(pkv) -> Cache:
    """
    Convert whatever the model returned into our plain tuple form.

    The cache object has changed shape three times across transformers
    versions, so this handles all of them:

      <= 4.46   a plain tuple of (keys, values) per layer
      4.47-4.53 a DynamicCache with .to_legacy_cache() and .key_cache
      >= 4.54   a DynamicCache with .layers[i].keys / .layers[i].values,
                not subscriptable, no legacy methods at all (this is what
                transformers 5.x gives you)

    Everything downstream works on the tuple, so version churn stops here.
    """
    if pkv is None:
        return None

    # already a plain tuple
    if isinstance(pkv, (tuple, list)):
        return tuple(pkv)

    # 4.47 - 4.53
    if hasattr(pkv, "to_legacy_cache"):
        try:
            return tuple(pkv.to_legacy_cache())
        except Exception:
            pass

    # 4.54+ / 5.x
    if hasattr(pkv, "layers"):
        return tuple((layer.keys, layer.values) for layer in pkv.layers)

    # 4.47 - 4.53 internals
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return tuple(zip(pkv.key_cache, pkv.value_cache))

    raise TypeError(
        f"unrecognised cache type {type(pkv).__name__}. "
        "Add a branch to normalise() for this transformers version."
    )


def _dynamic_cache_cls():
    """The DynamicCache class, or None on very old transformers."""
    try:
        from transformers.cache_utils import DynamicCache
        return DynamicCache
    except Exception:
        return None


def denormalise(cache: Cache | None):
    """
    Convert our tuple back into whatever the installed model expects.

    Newer transformers refuses a plain tuple for past_key_values, so it
    has to be wrapped. Wrapping shares the same tensors - nothing is
    copied - and the model's own update() concatenates into fresh
    tensors, so our tuple is never mutated behind our back.
    """
    if cache is None:
        return None

    DynamicCache = _dynamic_cache_cls()
    if DynamicCache is None:
        return cache                      # old transformers takes tuples

    if hasattr(DynamicCache, "from_legacy_cache"):
        try:
            return DynamicCache.from_legacy_cache(cache)
        except Exception:
            pass

    try:
        dc = DynamicCache()
    except TypeError:
        dc = DynamicCache(config=None)
    for layer_idx, (k, v) in enumerate(cache):
        dc.update(k, v, layer_idx)
    return dc


def shares_memory(a: Cache, b: Cache) -> bool:
    """True if the two caches point at the same underlying storage."""
    return a[0][0].data_ptr() == b[0][0].data_ptr()


def max_diff(a: Cache, b: Cache) -> float:
    """
    Largest absolute difference between two caches, across every layer.
    Used by the tests: 0.0 means bit-identical.
    """
    n = min(length(a), length(b))
    worst = 0.0
    for layer in range(n_layers(a)):
        for side in (0, 1):                      # keys, then values
            d = (a[layer][side][:, :, :n, :].float()
                 - b[layer][side][:, :, :n, :].float()).abs().max().item()
            worst = max(worst, d)
    return worst


# ======================================================================
# cache primitives
# ======================================================================

def fork(cache: Cache, label: str = "") -> Cache:
    """
    Independent copy of a cache.

    This is how several suspects share one prefilled dossier without
    re-reading it, and how stage 5 tries a question without committing.
    The copy shares no memory with the original.
    """
    out = copy.deepcopy(cache)
    COUNTER.event("fork", length(cache), label)
    return out


def crop(cache: Cache, n: int, label: str = "") -> Cache:
    """
    Return a cache holding only the first n tokens. This is the rewind.

    Clones rather than slicing, because a slice shares memory with its
    source.

    Correctness: attention is causal, so tokens after position n never
    influenced tokens before it. A cropped cache is therefore bit-identical
    to a fresh prefill of the first n tokens, not an approximation.
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
    transfer on the way back. Stage 6's eviction policy picks the device
    per snapshot.
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


# ======================================================================
# running the model
# ======================================================================

def prefill(model, ids: torch.Tensor, cache: Cache | None = None,
            label: str = "", count_naive: bool = False):
    """
    Push `ids` through the model, optionally continuing from `cache`.
    Returns (new_cache, logits).

    Two things must be right, and both fail silently if wrong:
      - attention_mask covers cached tokens PLUS the new ones
      - position_ids continue from where the cache ended, not from 0.
        RoPE rotates each key by its position, so wrong positions mean
        the model computes wrong distances between tokens.
    """
    device = next(model.parameters()).device
    n = length(cache)
    m = ids.shape[1]

    with torch.no_grad():
        out = model(
            input_ids=ids,
            past_key_values=denormalise(cache),
            attention_mask=torch.ones(1, n + m, device=device),
            position_ids=torch.arange(n, n + m, device=device).unsqueeze(0),
            use_cache=True,
        )

    COUNTER.prefill(m, label)
    if count_naive:
        COUNTER.naive(n + m, label)

    return normalise(out.past_key_values), out.logits


def _pick_token(logits, temperature, generator=None):
    """
    Choose the next token.

    temperature == 0  -> greedy (argmax). Fully deterministic, which is
                         what stage 5 needs for reproducible scores.
    temperature > 0   -> sample from the softened distribution. Stage 7
                         raises this as a suspect's composure drops.
    """
    last = logits[:, -1, :]
    if temperature <= 0.0:
        return last.argmax(-1, keepdim=True)
    probs = torch.softmax(last.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def decode(model, tokenizer, cache: Cache, logits: torch.Tensor,
           max_tokens: int = 40, temperature: float = 0.0,
           seed: int | None = None, stop_ids: tuple | None = None):
    """
    Generate from the current cache, one token at a time.

    Each step feeds exactly ONE token, because the cache already holds
    everything before it.

    Returns (text, new_cache, produced_ids). The ids matter: re-tokenising
    the decoded text does NOT reliably round-trip to the same token count,
    so callers tracking what the cache holds must use these, not a
    re-tokenisation of the string.
    """
    device = next(model.parameters()).device

    generator = None
    if temperature > 0.0 and seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    if stop_ids is None:
        candidates = [tokenizer.eos_token_id]
        for marker in ("<|im_end|>", "<|endoftext|>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(marker)
            except Exception:
                tid = None
            if tid is not None and tid >= 0:
                candidates.append(tid)
        stop_ids = tuple(t for t in candidates if t is not None)

    nxt = _pick_token(logits, temperature, generator)
    produced: list[int] = []

    for _ in range(max_tokens):
        if nxt.item() in stop_ids:
            break
        produced.append(nxt.item())

        n = length(cache)
        with torch.no_grad():
            out = model(
                input_ids=nxt,
                past_key_values=denormalise(cache),
                attention_mask=torch.ones(1, n + 1, device=device),
                position_ids=torch.tensor([[n]], device=device),
                use_cache=True,
            )
        cache = normalise(out.past_key_values)
        nxt = _pick_token(out.logits, temperature, generator)

    COUNTER.decode(len(produced))
    return tokenizer.decode(produced), cache, produced