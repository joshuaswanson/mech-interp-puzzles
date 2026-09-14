import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
from common import *

model = load_model()
E = model.tok_embed.weight
W_U = model.unembed.weight
IDX = torch.arange(1, 11).float()
IDX_C = IDX - IDX.mean()


def count_slope(v):
    return ((v @ W_U[COUNT_BASE:].T) * IDX_C).sum(-1) / (IDX_C ** 2).sum()


def run(x):
    b, s = x.shape
    h = E[x]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    comps = {}
    for L, layer in enumerate(model.layers):
        outs = []
        for hi, head in enumerate(layer.heads):
            o, a = head(h, mask)
            outs.append(o)
            comps[f"L{L}H{hi}"] = o @ layer.W_O.weight[:, hi * D_HEAD:(hi + 1) * D_HEAD].T
        h = h + layer.W_O(torch.cat(outs, -1))
    return comps, h


rng = np.random.default_rng(0)
X, C = random_batch(rng, 300)
comps, final = run(X)
P = torch.zeros(len(X), 10)
for b in range(len(X)):
    P[b, X[b, 1:-1]] = 1

# count direction fit and unembed quadratic
X1 = torch.stack([C.float(), torch.ones(len(C))], 1)
W = torch.linalg.lstsq(X1, final[:, -1]).solution
g, r0 = W[0], W[1]
A = torch.stack([IDX ** 2, IDX, torch.ones(10)], 1)
cg = torch.linalg.lstsq(A, (W_U[COUNT_BASE:] @ g).unsqueeze(1)).solution.squeeze()
cr = torch.linalg.lstsq(A, (W_U[COUNT_BASE:] @ r0).unsqueeze(1)).solution.squeeze()
unit = cg[1].item()

# per-head intercept slopes from the presence regression
Xp = torch.cat([P, torch.ones(len(P), 1)], 1)
intercepts = []
for hi in range(4):
    Y = comps[f"L1H{hi}"][:, -1] @ W_U[COUNT_BASE:].T
    Wp = torch.linalg.lstsq(Xp, Y).solution
    bias = Wp[-1]
    intercepts.append((((IDX_C * (bias - bias.mean())).sum() / (IDX_C ** 2).sum()).item()))

out = {
    "E": E.tolist(),
    "U": W_U.tolist(),
    "layers": [
        {
            "heads": [{"WQ": h.W_Q.weight.tolist(), "WK": h.W_K.weight.tolist(), "WV": h.W_V.weight.tolist()} for h in layer.heads],
            "WO": layer.W_O.weight.tolist(),
        }
        for layer in model.layers
    ],
    "constants": {
        "unit": unit,
        "intercept_slopes": intercepts,
        "quad": {"a": cr[0].item(), "b": cr[1].item(), "lin": cg[1].item()},
        "v": g.tolist(),
        "r0": r0.tolist(),
    },
}
json.dump(out, open(HERE / "model_weights.json", "w"))

# reference outputs for the JS test
ref = []
rng = np.random.default_rng(123)
for c in range(1, 11):
    s = random_sequence(rng, c)
    x = torch.tensor([[BOS] + s + [ANS]])
    _, h = run(x)
    logits, attns = model(x)
    ref.append({
        "tokens": x[0].tolist(),
        "logits": logits[0, -1, COUNT_BASE:].tolist(),
        "attn_l1_ans": attns[1][0, :, -1, :].tolist(),
        "attn_l0_row5": attns[0][0, :, 5, :].tolist(),
    })
json.dump(ref, open(HERE / "reference_outputs.json", "w"))
print("unit", round(unit, 2), "intercepts", [round(v, 1) for v in intercepts], "quad", {k: round(v, 2) for k, v in out["constants"]["quad"].items()})
