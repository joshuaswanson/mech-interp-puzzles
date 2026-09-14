import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

torch.set_grad_enabled(False)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
np.set_printoptions(precision=3, suppress=True, linewidth=200)

BOS, SEP, ANS, EOS = 10, 11, 12, 13
NAMES = {10: "BOS", 11: "SEP", 12: "ANS", 13: "EOS"}


def load(which):
    repo = f"andyrdt/04_2026_puzzle_1{which}"
    p = hf_hub_download(repo, "model.py")
    spec = importlib.util.spec_from_file_location(f"model_apr_{which}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    model = mod.AttentionOnlyTransformer.from_config(config["model"])
    model.load_state_dict(torch.load(hf_hub_download(repo, "model.pt"), map_location="cpu", weights_only=True))
    model.eval()
    return model


def label(t):
    return NAMES.get(t, str(t))


def tok1(nums):
    out = [BOS]
    for i, n in enumerate(nums):
        out.append(n)
        if i < len(nums) - 1: out.append(SEP)
    out.append(ANS)
    return out


def tok2(nums):
    out = [BOS]
    for i, n in enumerate(nums):
        out += [n // 10, n % 10]
        if i < len(nums) - 1: out.append(SEP)
    out.append(ANS)
    return out
