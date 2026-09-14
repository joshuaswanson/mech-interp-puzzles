import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight
U = model.unembed.weight[COUNT_BASE:]

def run_all(x):
    b, s = x.shape
    h = E[x]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    store = {"embed": h.clone()}
    for L, layer in enumerate(model.layers):
        outs = []
        for hi, head in enumerate(layer.heads):
            o, a = head(h, mask)
            outs.append(o)
            store[f"L{L}H{hi}"] = (o @ layer.W_O.weight[:, hi * D_HEAD:(hi + 1) * D_HEAD].T)
        h = h + layer.W_O(torch.cat(outs, -1))
        store[f"resid_L{L}"] = h.clone()
    return store

rng = np.random.default_rng(7)
x, counts = random_batch(rng, 400)
st = run_all(x)
B = len(counts)
presence = torch.zeros(B, 10)
for b in range(B):
    for t in x[b, 1:-1].tolist():
        presence[b, t] = 1

def fit(X, Y):
    X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
    W = torch.linalg.lstsq(X1, Y).solution
    return W[:-1], W[-1]

# The unembed rows for counts 1..10. Look for the "count direction" g and "bias" structure.
# Fit final logits ~ presence.
final = st["resid_L1"][:, -1] @ U.T
W, bias = fit(presence, final)
print("Final logits ~ presence:  each row = logit vector (over counts #1..#10) added when symbol k is present")
print("        " + " ".join(f"#{c:<5d}" for c in range(1, 11)))
for k in range(10):
    print(f"  {chr(97+k)}:    " + " ".join(f"{v:6.1f}" for v in W[k].tolist()))
print(f"  bias: " + " ".join(f"{v:6.1f}" for v in bias.tolist()))

# Check: is W[k] roughly linear in count index for each k? fit slope
idx = torch.arange(1, 11).float()
print("\nSlope of each symbol's logit-vector vs count index (should be similar across symbols if 'count direction' is shared):")
for k in range(10):
    w = W[k]
    slope = ((idx - idx.mean()) * (w - w.mean())).sum() / ((idx - idx.mean()) ** 2).sum()
    print(f"  {chr(97+k)}: slope = {slope:6.2f}")
print("bias vector fit to quadratic a*c^2 + b*c + d:")
A = torch.stack([idx**2, idx, torch.ones(10)], 1)
coef = torch.linalg.lstsq(A, bias.unsqueeze(1)).solution.squeeze()
print(f"  a={coef[0]:.2f} b={coef[1]:.2f} d={coef[2]:.2f}")

print("\n\nPer-head presence coefficients, summarized by 'slope' (how much each present symbol pushes toward higher counts):")
print(f"{'head':6s} " + " ".join(f"{chr(97+k):>6s}" for k in range(10)) + "   bias-slope")
for name in ["embed", "L1H0", "L1H1", "L1H2", "L1H3"]:
    Y = st[name][:, -1] @ U.T
    W, bias = fit(presence, Y)
    slopes = []
    for k in range(10):
        w = W[k]
        slopes.append((((idx - idx.mean()) * (w - w.mean())).sum() / ((idx - idx.mean()) ** 2).sum()).item())
    bs = (((idx - idx.mean()) * (bias - bias.mean())).sum() / ((idx - idx.mean()) ** 2).sum()).item()
    print(f"{name:6s} " + " ".join(f"{s:6.2f}" for s in slopes) + f"   {bs:6.2f}")
