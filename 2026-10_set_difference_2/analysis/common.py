import importlib.util
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

torch.set_grad_enabled(False)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=220)
np.set_printoptions(precision=3, suppress=True, linewidth=220)

REPO_ID = "andyrdt/10_2026_puzzle_1"
NUM_SYMBOLS, BOS, SEP, EOS = 16, 16, 17, 18
SYM = [chr(97 + k) for k in range(NUM_SYMBOLS)]


def load_model():
    p = hf_hub_download(REPO_ID, "model.py")
    spec = importlib.util.spec_from_file_location("model_oct", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = json.loads(Path(hf_hub_download(REPO_ID, "config.json")).read_text())
    model = mod.Transformer.from_config(config["model"])
    model.load_state_dict(torch.load(hf_hub_download(REPO_ID, "model.pt"), map_location="cpu", weights_only=True))
    model.eval()
    return model, config, mod


def label(t):
    return SYM[t] if t < NUM_SYMBOLS else {BOS: "BOS", SEP: "SEP", EOS: "EOS"}[t]


def encode(X, Y):
    return [BOS] + list(X) + [SEP] + list(Y) + [EOS]


def make_prompt(rng, K):
    X = rng.sample(range(NUM_SYMBOLS), K)
    Y = X[:]
    rng.shuffle(Y)
    return encode(X, Y)


def scored_positions(tokens):
    """Positions SEP..yK and the set of valid next tokens at each."""
    K = (len(tokens) - 3) // 2
    X = tokens[1:1 + K]
    out = {}
    for pos in range(K + 1, 2 * K + 2):
        seen = set(tokens[K + 2:pos + 1])
        remaining = [s for s in X if s not in seen]
        out[pos] = remaining if remaining else [EOS]
    return out


def batch(rng, n_per_K, Ks=range(2, 9)):
    """Group by K so every sequence in a batch has the same length."""
    out = {}
    for K in Ks:
        out[K] = torch.tensor([make_prompt(rng, K) for _ in range(n_per_K)])
    return out
