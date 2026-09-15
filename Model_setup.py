"""
Model_setup.py
--------------
One place that loads the model, so every script does it the same way and
version differences are handled once.

CHOOSING A MODEL
----------------
Answer quality tracks model size closely here, and it is the difference
between a demo of mechanics and a game worth playing:

    0.5B   echoes the question back with the pronouns flipped, loses the
           persona after a turn or two. Fine for testing the cache.
    1.5B   holds a character, stops echoing, still repetitive over a long
           session. The most this project can run natively on 4 GB.
    3B     maintains a lie under pressure. Needs ~6.2 GB at fp16, so on a
           4 GB card it has to spill layers to system RAM - it works, but
           expect several seconds a turn.
    7-8B   deflects, stays in character, actually fun. Cloud only.

The usual escapes for fitting a big model in a small card - bitsandbytes
4-bit, GPTQ, AWQ - all want compute capability 7.5 or newer. A Quadro
P2000 is 6.1, so none of them apply. That is why the choice below is
between fp16 and CPU offload rather than quantisation.

    set SANDBOX_MODEL=Qwen/Qwen2.5-1.5B-Instruct     pick explicitly
    set SANDBOX_OFFLOAD=1                            allow CPU spill
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# name -> approximate fp16 weight size in GB
REGISTRY = {
    "Qwen/Qwen2.5-0.5B-Instruct": 1.0,
    "Qwen/Qwen2.5-1.5B-Instruct": 3.1,
    "Qwen/Qwen2.5-3B-Instruct":   6.2,
    "Qwen/Qwen2.5-7B-Instruct":  15.2,
    "meta-llama/Llama-3.2-1B-Instruct": 2.5,
    "meta-llama/Llama-3.2-3B-Instruct": 6.4,
}

# best first - pick the largest that fits
PREFERENCE = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

# leave room for the KV caches, which is the whole point of the project
HEADROOM_GB = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def pick_model(budget_gb=None, allow_offload=None):
    """
    The largest model that fits, or an explicit override.

    Returns (name, offload) where offload=True means layers will spill to
    system RAM. That is slow but it does run, and it is the only way to
    get past 1.5B on a 4 GB card.
    """
    forced = os.environ.get("SANDBOX_MODEL", "").strip()
    if forced:
        return forced, os.environ.get("SANDBOX_OFFLOAD", "") == "1"

    if allow_offload is None:
        allow_offload = os.environ.get("SANDBOX_OFFLOAD", "") == "1"

    budget = (budget_gb if budget_gb is not None else vram_gb()) - HEADROOM_GB

    for name in PREFERENCE:
        if REGISTRY.get(name, 999) <= budget:
            return name, False

    if allow_offload:
        # one step above what fits, spilled to CPU
        for name in PREFERENCE:
            if REGISTRY.get(name, 999) <= budget * 2.2:
                return name, True

    return PREFERENCE[-1], False


MODEL_NAME, _OFFLOAD = pick_model()


def load_model(name=None, device=None, quiet=False, offload=None):
    """
    Returns (tokenizer, model, device).

    eager attention on purpose: it is the plain, reproducible attention
    path, and the tests compare caches numerically.
    """
    if name is None:
        name, auto_offload = pick_model()
        offload = auto_offload if offload is None else offload
    offload = bool(offload)
    device = device or DEVICE

    if not quiet:
        size = REGISTRY.get(name)
        where = f"{device}{' + cpu offload' if offload else ''}"
        print(f"loading {name} on {where} ..."
              + (f"  (~{size:.1f} GB fp16, card has {vram_gb():.1f} GB)"
                 if size and device == "cuda" else ""))

    tok = AutoTokenizer.from_pretrained(name)
    dt = torch.float16 if device == "cuda" else torch.float32

    kwargs = {"attn_implementation": "eager"}
    if offload and device == "cuda":
        free = max(1.0, vram_gb() - HEADROOM_GB)
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: f"{free:.1f}GiB", "cpu": "24GiB"}

    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dt, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt, **kwargs)

    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()

    if not quiet:
        p = sum(x.numel() for x in model.parameters()) / 1e9
        print(f"  {p:.2f}B params, {model.config.num_hidden_layers} layers, "
              f"{model.config.num_key_value_heads} kv heads")
        if device == "cuda" and not offload:
            used = torch.cuda.memory_allocated() / 1e9
            print(f"  {used:.2f} GB weights resident, "
                  f"{vram_gb() - used:.2f} GB left for caches")

    return tok, model, device


def tolerance(model):
    """
    How close is "the same" when comparing two caches.

    Pure tensor operations - fork, crop, snapshot - are bit-identical and
    must compare exactly 0.0.

    Anything that RECOMPUTES through the model can differ slightly when
    the matmul shapes differ: prefilling 7 tokens onto 14 cached is not
    the same sequence of floating point reductions as prefilling 21 at
    once. Measured on real Qwen2.5-0.5B: ~9e-5 in fp32 on CPU. The limits
    below sit well above observed noise but far below any real error,
    which would be order-of-magnitude, not fifth-decimal.
    """
    dt = next(model.parameters()).dtype
    if dt == torch.float16:
        return 5e-2
    if dt == torch.bfloat16:
        return 2e-1
    return 1e-3


if __name__ == "__main__":
    print(f"cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"card: {torch.cuda.get_device_name(0)}  "
              f"{vram_gb():.1f} GB  cc {torch.cuda.get_device_capability(0)}")
    print()
    for budget in (4, 6, 8, 12, 16, 24):
        fits, off = pick_model(budget_gb=budget)
        print(f"  {budget:2d} GB -> {fits}{'  (cpu offload)' if off else ''}")
    print()
    chosen, off = pick_model()
    print(f"this machine -> {chosen}{'  (cpu offload)' if off else ''}")
    print("\noverride with:  set SANDBOX_MODEL=Qwen/Qwen2.5-3B-Instruct")
    print("allow spilling: set SANDBOX_OFFLOAD=1")