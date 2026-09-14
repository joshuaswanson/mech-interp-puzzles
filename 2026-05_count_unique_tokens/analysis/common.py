import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

torch.set_grad_enabled(False)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
np.set_printoptions(precision=3, suppress=True, linewidth=200)

REPO_ID = "andyrdt/05_2026_puzzle_1"
NUM_SYMBOLS = 10
SEQ_LEN = 10
BOS = 10
ANS = 11
COUNT_BASE = 12
VOCAB_SIZE = 22
D_HEAD = 8


def load_model():
    model_py_path = hf_hub_download(REPO_ID, "model.py")
    spec = importlib.util.spec_from_file_location("model", model_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = json.loads(Path(hf_hub_download(REPO_ID, "config.json")).read_text())
    model = mod.AttentionOnlyTransformer.from_config(config["model"])
    model.load_state_dict(torch.load(hf_hub_download(REPO_ID, "model.pt"), map_location="cpu", weights_only=True))
    model.eval()
    return model


def token_label(tok):
    if 0 <= tok < NUM_SYMBOLS:
        return chr(ord("a") + tok)
    if tok == BOS:
        return "BOS"
    if tok == ANS:
        return "ANS"
    if COUNT_BASE <= tok < COUNT_BASE + SEQ_LEN:
        return f"#{tok - COUNT_BASE + 1}"
    return f"?{tok}"


def encode(symbols):
    ids = [ord(s) - ord("a") if isinstance(s, str) else int(s) for s in symbols]
    return [BOS] + ids + [ANS]


def random_sequence(rng, c):
    symbols = rng.choice(NUM_SYMBOLS, size=c, replace=False)
    extras = rng.choice(symbols, size=SEQ_LEN - c, replace=True)
    seq = np.concatenate([symbols, extras])
    rng.shuffle(seq)
    return seq.tolist()


def random_batch(rng, n_per_count):
    seqs, counts = [], []
    for c in range(1, SEQ_LEN + 1):
        for _ in range(n_per_count):
            s = random_sequence(rng, c)
            seqs.append([BOS] + s + [ANS])
            counts.append(c)
    return torch.tensor(seqs), torch.tensor(counts)


def predict(model, x):
    logits, attns = model(x)
    return logits[:, -1, COUNT_BASE:COUNT_BASE + SEQ_LEN].argmax(-1) + 1, logits, attns
