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

rng = np.random.default_rng(6)
x, counts = random_batch(rng, 400)
st = run_all(x)
B = len(counts)
presence = torch.zeros(B, 10)
mult = torch.zeros(B, 10)
for b in range(B):
    for t in x[b, 1:-1].tolist():
        presence[b, t] = 1
        mult[b, t] += 1

def lstsq_r2(X, y):
    X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
    w = torch.linalg.lstsq(X1, y.float().unsqueeze(1)).solution
    pred = (X1 @ w).squeeze(1)
    return 1 - ((pred - y) ** 2).mean() / y.float().var(), pred

print("Linear probe R^2 for count from each activation at ANS position:")
for name in ["embed", "L0H0", "L0H1", "L0H2", "L0H3", "resid_L0", "L1H0", "L1H1", "L1H2", "L1H3", "resid_L1"]:
    r2, pred = lstsq_r2(st[name][:, -1], counts.float())
    acc = (pred.round() == counts).float().mean()
    print(f"  {name:9s}: R^2 = {r2:.3f}   rounded-acc = {acc:.3f}")

print("\nHow much does L0_out[ANS] depend on presence vs multiplicity?")
L0ans = st["resid_L0"][:, -1] - st["embed"][:, -1]
X = torch.cat([presence, mult], 1)
for k in range(32):
    pass
r2p, _ = lstsq_r2(presence, L0ans[:, 0]); 
# fit each dim separately, report avg R^2 for presence-only vs presence+mult
def multi_r2(X, Y):
    X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
    W = torch.linalg.lstsq(X1, Y).solution
    P = X1 @ W
    return 1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
print(f"  R^2 from presence only     : {multi_r2(presence, L0ans):.3f}")
print(f"  R^2 from multiplicity only : {multi_r2(mult, L0ans):.3f}")
print(f"  R^2 from both              : {multi_r2(torch.cat([presence, mult], 1), L0ans):.3f}")
print(f"  R^2 from count only        : {multi_r2(counts.float().unsqueeze(1), L0ans):.3f}")

print("\nSame for final resid at ANS (logit-relevant):")
R = st["resid_L1"][:, -1] @ U.T
print(f"  logits R^2 from presence   : {multi_r2(presence, R):.3f}")
print(f"  logits R^2 from count      : {multi_r2(counts.float().unsqueeze(1), R):.3f}")
print(f"  logits R^2 from count+count^2: {multi_r2(torch.stack([counts.float(), counts.float()**2], 1), R):.3f}")

print("\nPer-head logit contribution: R^2 from count, count^2 and from presence")
for name in ["L1H0", "L1H1", "L1H2", "L1H3"]:
    R = st[name][:, -1] @ U.T
    c = counts.float()
    print(f"  {name}: count: {multi_r2(c.unsqueeze(1), R):.3f}  count+c^2: {multi_r2(torch.stack([c, c**2], 1), R):.3f}  presence: {multi_r2(presence, R):.3f}  presence+count: {multi_r2(torch.cat([presence, c.unsqueeze(1)],1), R):.3f}")
