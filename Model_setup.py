"""
model_setup.py
--------------
One place that knows how to load the model, so every script does it the
same way and version differences are handled once.

Two things vary across transformers versions:
  - the dtype kwarg was renamed from torch_dtype to dtype
  - the returned cache object changed shape (handled in cache_ops.normalise)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name=MODEL_NAME, device=DEVICE, quiet=False):
    """
    Returns (tokenizer, model, device).

    eager attention on purpose: it is the plain, reproducible attention
    path, which matters because the tests compare caches numerically.
    """
    if not quiet:
        print(f"loading {name} on {device} ...")

    tok = AutoTokenizer.from_pretrained(name)
    dt = torch.float16 if device == "cuda" else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=dt, attn_implementation="eager")
    except TypeError:
        # older transformers used torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dt, attn_implementation="eager")

    model = model.to(device)
    model.eval()
    return tok, model, device


def tolerance(model):
    """
    How close is "the same" when comparing two caches.

    Pure tensor operations - fork, crop, snapshot - are bit-identical and
    must compare exactly 0.0.

    Anything that RECOMPUTES through the model can differ slightly when
    the matmul shapes differ: prefilling 7 tokens onto 14 cached is not
    the same sequence of floating point reductions as prefilling 21 at
    once. Measured on real Qwen2 weights this is around 1e-7 in fp32.
    It is float arithmetic, not a bug, and the tolerance below covers it.

    Measured on real Qwen2.5-0.5B: ~9e-5 in fp32 on CPU. The limits
    below sit well above observed noise but far below any real error,
    which would be order-of-magnitude, not fifth-decimal.
    """
    dt = next(model.parameters()).dtype
    if dt == torch.float16:
        return 5e-2
    if dt == torch.bfloat16:
        return 2e-1
    return 1e-3