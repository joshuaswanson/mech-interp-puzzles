import importlib.util
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

torch.set_grad_enabled(False)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
np.set_printoptions(precision=3, suppress=True, linewidth=200)

REPO_ID = "andyrdt/09_2026_puzzle_1"
NUM_SYMBOLS, SET_SIZE, BOS, SEP = 16, 4, 16, 17


def load_model():
    p = hf_hub_download(REPO_ID, "model.py")
    spec = importlib.util.spec_from_file_location("model_sep", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = json.loads(Path(hf_hub_download(REPO_ID, "config.json")).read_text())
    model = mod.AttentionOnlyTransformer.from_config(config["model"])
    model.load_state_dict(torch.load(hf_hub_download(REPO_ID, "model.pt"), map_location="cpu", weights_only=True))
    model.eval()
    return model


def label(t):
    return chr(97 + t) if t < NUM_SYMBOLS else ("BOS" if t == BOS else "SEP")


def encode(X, Y):
    return [BOS] + list(X) + [SEP] + list(Y) + [SEP]


def all_prompts(max_n=None, seed=0):
    """Every (X, z) with every ordering, or a random subset."""
    rng = random.Random(seed)
    out = []
    for X in itertools.combinations(range(NUM_SYMBOLS), SET_SIZE):
        for z in X:
            Y = [s for s in X if s != z]
            for px in itertools.permutations(X):
                for py in itertools.permutations(Y):
                    out.append((encode(px, py), z))
    if max_n is not None and len(out) > max_n:
        out = rng.sample(out, max_n)
    return out
