"""
Snapshots.py   -  Stage 6
-------------------------
A snapshot store with a memory budget and three storage tiers.

WHY THIS EXISTS
---------------
Every turn saves a snapshot so it can be rewound to. Three suspects,
a dozen turns each, plus three speculative forks per turn, and the
snapshots stop fitting on a 4 GB card. At 12 KB per token an 800-token
snapshot is about 10 MB, so a hundred of them is a gigabyte.

So the store has to decide what to keep where. Four states, ordered by
how expensive it is to get a snapshot back:

    GPU      full precision, resident. Instant.
    INT8     quantised, still on GPU. Half the bytes, needs a dequantise
             on the way out, and is slightly lossy.
    CPU      full precision in host RAM. Lossless, costs a PCIe transfer.
    DROPPED  gone. The caller has to recompute it from scratch.

Demotion is least-recently-used: when the GPU budget is exceeded, the
snapshot nobody has touched for longest moves down a tier.

This is the stage where the 4 GB limit stops being an apology and starts
being the reason the problem is interesting.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

import cache_ops as co

GPU = "gpu"
INT8 = "int8"
CPU = "cpu"
DROPPED = "dropped"

TIER_ORDER = [GPU, INT8, CPU, DROPPED]


# ======================================================================
# int8 quantisation
# ======================================================================

def quantise(cache):
    """
    Symmetric per-tensor int8.

    For each layer's keys and values separately, find the largest
    magnitude, scale it to fit 127, and round. Storing the scale lets us
    undo it later.

    Per-tensor is the simplest scheme and the easiest to explain. Keys
    are known to have outlier channels, so a per-channel scale would lose
    less; that is a worthwhile experiment for stage 12, not a requirement
    here. The test measures the actual error either way.
    """
    out = []
    for k, v in cache:
        ks = k.abs().amax().clamp(min=1e-8) / 127.0
        vs = v.abs().amax().clamp(min=1e-8) / 127.0
        out.append((
            (k / ks).round().clamp(-127, 127).to(torch.int8), ks,
            (v / vs).round().clamp(-127, 127).to(torch.int8), vs,
        ))
    return tuple(out)


def dequantise(qcache, dtype=torch.float16, device="cuda"):
    """Undo quantise(). Lossy - the error is measured in the tests."""
    return tuple(
        ((qk.to(device).to(torch.float32) * ks.to(device)).to(dtype),
         (qv.to(device).to(torch.float32) * vs.to(device)).to(dtype))
        for qk, ks, qv, vs in qcache
    )


def qbytes(qcache):
    """Bytes held by a quantised cache, scales included."""
    total = 0
    for qk, ks, qv, vs in qcache:
        total += qk.numel() * qk.element_size() + qv.numel() * qv.element_size()
        total += ks.numel() * ks.element_size() + vs.numel() * vs.element_size()
    return total


# ======================================================================
# entries
# ======================================================================

@dataclass
class Entry:
    key: str
    tier: str
    data: object                 # cache tuple, or quantised form, or None
    n_tokens: int
    nbytes: int
    last_used: int
    dtype: object = None

    def describe(self):
        return f"{self.key}[{self.tier}] {self.n_tokens}tok {self.nbytes/1e6:.1f}MB"


@dataclass
class StoreStats:
    puts: int = 0
    gets: int = 0
    hits_gpu: int = 0
    hits_int8: int = 0
    hits_cpu: int = 0
    misses: int = 0              # asked for a dropped snapshot
    demotions: dict = field(default_factory=lambda: {INT8: 0, CPU: 0, DROPPED: 0})

    @property
    def hit_rate(self):
        served = self.hits_gpu + self.hits_int8 + self.hits_cpu
        return served / self.gets if self.gets else 0.0

    def summary(self):
        return (f"puts={self.puts} gets={self.gets} "
                f"hits(gpu/int8/cpu)={self.hits_gpu}/{self.hits_int8}/{self.hits_cpu} "
                f"misses={self.misses} hit_rate={self.hit_rate:.0%} "
                f"demotions={self.demotions[INT8]}/{self.demotions[CPU]}/"
                f"{self.demotions[DROPPED]}")


# ======================================================================
# the store
# ======================================================================

class SnapshotStore:
    """
    Keyed snapshot storage under a GPU byte budget.

    gpu_budget_bytes=None means unlimited, which is the old behaviour and
    the default so nothing breaks if you do not opt in.
    """

    def __init__(self, gpu_budget_bytes=None, device="cuda",
                 use_int8=True, dtype=torch.float16):
        self.budget = gpu_budget_bytes
        self.device = device
        self.use_int8 = use_int8
        self.dtype = dtype
        self.entries: dict[str, Entry] = {}
        self.stats = StoreStats()
        self._clock = itertools.count()

    # -------------------------------------------------- bookkeeping
    def _touch(self, entry):
        entry.last_used = next(self._clock)

    def gpu_bytes(self):
        """How many bytes we are currently holding on the GPU."""
        return sum(e.nbytes for e in self.entries.values()
                   if e.tier in (GPU, INT8))

    def tier_counts(self):
        counts = {t: 0 for t in TIER_ORDER}
        for e in self.entries.values():
            counts[e.tier] += 1
        return counts

    # -------------------------------------------------- put / get
    def put(self, key, cache):
        """Store a snapshot at the top tier, then evict if over budget."""
        self.entries[key] = Entry(
            key=key, tier=GPU, data=co.fork(cache, label=f"snap:{key}"),
            n_tokens=co.length(cache), nbytes=co.nbytes(cache),
            last_used=next(self._clock), dtype=cache[0][0].dtype,
        )
        self.stats.puts += 1
        co.COUNTER.event("snapshot", co.length(cache), f"{key}->gpu")
        self._enforce_budget()
        return key

    def get(self, key):
        """
        Bring a snapshot back to the device.

        Returns a cache tuple, or None if it was dropped and must be
        recomputed by the caller.
        """
        self.stats.gets += 1
        entry = self.entries.get(key)

        if entry is None or entry.tier == DROPPED:
            self.stats.misses += 1
            co.COUNTER.event("restore", 0, f"{key}:MISS")
            return None

        if entry.tier == GPU:
            self.stats.hits_gpu += 1
            cache = co.fork(entry.data, label=f"restore:{key}")
        elif entry.tier == INT8:
            self.stats.hits_int8 += 1
            cache = dequantise(entry.data, self.dtype, self.device)
        else:                                   # CPU
            self.stats.hits_cpu += 1
            cache = co.restore(entry.data, device=self.device,
                               label=f"{key}:cpu->gpu")

        self._touch(entry)
        co.COUNTER.event("restore", entry.n_tokens, f"{key}<-{entry.tier}")
        return cache

    def forget(self, key):
        self.entries.pop(key, None)

    # -------------------------------------------------- eviction
    def _enforce_budget(self):
        """Demote least-recently-used entries until we fit."""
        if self.budget is None:
            return
        guard = 0
        while self.gpu_bytes() > self.budget:
            guard += 1
            if guard > 10_000:
                raise RuntimeError("eviction loop did not converge")

            victim = self._lru_on_gpu()
            if victim is None:
                break                          # nothing left to demote
            self._demote(victim)

    def _lru_on_gpu(self):
        on_gpu = [e for e in self.entries.values() if e.tier in (GPU, INT8)]
        return min(on_gpu, key=lambda e: e.last_used) if on_gpu else None

    def _demote(self, entry):
        """Move one entry down exactly one tier."""
        if entry.tier == GPU and self.use_int8:
            q = quantise(entry.data)
            entry.data = q
            entry.tier = INT8
            entry.nbytes = qbytes(q)
            self.stats.demotions[INT8] += 1

        elif entry.tier in (GPU, INT8):
            if entry.tier == INT8:
                full = dequantise(entry.data, self.dtype, self.device)
            else:
                full = entry.data
            entry.data = co.snapshot(full, device="cpu", label=f"evict:{entry.key}")
            entry.tier = CPU
            entry.nbytes = co.nbytes(entry.data)
            self.stats.demotions[CPU] += 1

        else:
            entry.data = None
            entry.tier = DROPPED
            entry.nbytes = 0
            self.stats.demotions[DROPPED] += 1

        co.COUNTER.event("evict", entry.n_tokens, f"{entry.key}->{entry.tier}")

    def drop_coldest_cpu(self, n=1):
        """
        Free host RAM. CPU entries are not under the GPU budget, so this
        is called explicitly rather than automatically.
        """
        cpu = sorted((e for e in self.entries.values() if e.tier == CPU),
                     key=lambda e: e.last_used)
        for e in cpu[:n]:
            self._demote(e)

    # -------------------------------------------------- reporting
    def report(self):
        counts = self.tier_counts()
        return (f"gpu={counts[GPU]} int8={counts[INT8]} cpu={counts[CPU]} "
                f"dropped={counts[DROPPED]} "
                f"vram={self.gpu_bytes()/1e6:.1f}MB"
                + (f"/{self.budget/1e6:.0f}MB" if self.budget else ""))

    def table(self):
        rows = sorted(self.entries.values(), key=lambda e: -e.last_used)
        return "\n".join(f"  {e.describe()}" for e in rows)