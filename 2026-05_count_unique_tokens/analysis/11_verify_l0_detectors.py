import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight
U = model.unembed.weight[COUNT_BASE:]
idx = torch.arange(1, 11).float()
idx_c = idx - idx.mean()

def slope_readout(resid_vec):
    """Given residual-space vectors (..., 32), return the slope of the logit vector vs count index."""
    lg = resid_vec @ U.T
    return (lg * idx_c).sum(-1) / (idx_c ** 2).sum()

def layer0_out(x):
    b, s = x.shape
    h = E[x]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    layer = model.layers[0]
    outs, attns = zip(*[head(h, mask) for head in layer.heads])
    return layer.W_O(torch.cat(outs, -1)), torch.stack(attns, 1)

rng = np.random.default_rng(8)
x, counts = random_batch(rng, 300)
B = len(counts)
L0, A0 = layer0_out(x)
resid = E[x] + L0
seqs = x[:, 1:-1]

# Prefix presence: prefix_pres[b, j, k] = 1 if symbol k in x_1..x_j
prefix_pres = torch.zeros(B, SEQ_LEN, 10)
for b in range(B):
    seen = torch.zeros(10)
    for j in range(SEQ_LEN):
        seen[seqs[b, j]] = 1
        prefix_pres[b, j] = seen

# ---- L1H2: value at symbol position j, projected through W_O and unembed -> slope ----
head = model.layers[1].heads[2]
WO2 = model.layers[1].W_O.weight[:, 2 * D_HEAD:3 * D_HEAD]
val_resid = (resid[:, 1:-1] @ head.W_V.weight.T) @ WO2.T     # (B, 10, 32)
val_slope = slope_readout(val_resid)                          # (B, 10)
X = prefix_pres.reshape(-1, 10)
y = val_slope.reshape(-1)
X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
w = torch.linalg.lstsq(X1, y.unsqueeze(1)).solution.squeeze()
pred = X1 @ w
r2 = 1 - ((pred - y) ** 2).mean() / y.var()
print("L1H2 value at symbol position j (count-slope readout) ~ prefix presence of each symbol:")
print("  coeffs: " + " ".join(f"{chr(97+k)}={w[k]:6.2f}" for k in range(10)) + f"  bias={w[10]:.2f}  R^2={r2:.3f}")

# Which position does L1H2 attend to? Compare its attended position's prefix-presence of {a,d,g,h} to the full-sequence presence.
_, _, attns = predict(model, x)
A1 = attns[1][:, :, -1, :]
adgh = [0, 3, 6, 7]
full_adgh = prefix_pres[:, -1, adgh].sum(-1)
att_adgh = (A1[:, 2, 1:-1] * prefix_pres[:, :, adgh].sum(-1)).sum(-1) / A1[:, 2, 1:-1].sum(-1).clamp(min=1e-6)
print(f"\nL1H2: attention-weighted prefix count of {{a,d,g,h}} at attended positions vs full-sequence count:")
print(f"  mean |diff| = {(att_adgh - full_adgh).abs().mean():.3f};  fraction exact = {((att_adgh - full_adgh).abs() < 0.05).float().mean():.3f}")
# Attention mass on the last position / positions whose prefix already contains all of the {a,d,g,h} in the sequence
complete = (prefix_pres[:, :, adgh].sum(-1) == full_adgh.unsqueeze(1)).float()
print(f"  attention mass on positions whose prefix already has all {{a,d,g,h}} of the sequence: {(A1[:, 2, 1:-1] * complete).sum(-1).mean():.3f}")
print(f"  attention mass on the last symbol position: {A1[:, 2, 10].mean():.3f}")

# ---- L1H3: query from L0_out[ANS]; does L0_out[ANS] encode presence of {c,e,i,j}? ----
ceij = [2, 4, 8, 9]
full_pres = prefix_pres[:, -1]
# BOS attention of L1H3 vs # of {c,e,i,j}
n_ceij = full_pres[:, ceij].sum(-1)
n_other = full_pres.sum(-1) - n_ceij
print("\nL1H3 attention to BOS by (# of {c,e,i,j} present, # of other symbols present):")
print("           n_other: " + " ".join(f"{m:5d}" for m in range(0, 7)))
for n in range(0, 5):
    row = []
    for m in range(0, 7):
        sel = (n_ceij == n) & (n_other == m)
        row.append(f"{A1[sel, 3, 0].mean():5.2f}" if sel.sum() > 3 else "    -")
    print(f"  n_ceij={n}:         " + " ".join(row))

# L1H3 output slope vs n_ceij and n_other
head3 = model.layers[1].heads[3]
WO3 = model.layers[1].W_O.weight[:, 3 * D_HEAD:4 * D_HEAD]
out3 = (A1[:, 3].unsqueeze(-1) * (resid @ head3.W_V.weight.T)).sum(1) @ WO3.T
s3 = slope_readout(out3)
print("\nL1H3 output count-slope by (n_ceij, n_other):")
for n in range(0, 5):
    row = []
    for m in range(0, 7):
        sel = (n_ceij == n) & (n_other == m)
        row.append(f"{s3[sel].mean():6.1f}" if sel.sum() > 3 else "     -")
    print(f"  n_ceij={n}:         " + " ".join(row))

# Layer-0 heads at ANS: which symbol do they attend to (top-1) when present?
print("\nLayer-0 ANS-row attention: top-priority symbol per head (fraction of attention on it when present):")
for h in range(4):
    best = None
    for k in range(10):
        pres = full_pres[:, k] == 1
        m = (A0[pres, h, -1, 1:-1] * (seqs[pres] == k)).sum(-1).mean()
        if best is None or m > best[1]:
            best = (k, m.item())
    print(f"  L0H{h}: attends to '{chr(97+best[0])}' with total weight {best[1]:.2f} when present")
print("Layer-0 symbol-row attention (queries != b,f): top-priority symbol per head:")
for h in range(4):
    best = None
    for k in range(10):
        # positions j (query not b/f) whose prefix contains k
        qmask = (seqs != 1) & (seqs != 5) & (prefix_pres[:, :, k] == 1)
        att = A0[:, h, 1:-1, 1:-1]  # (B, 10q, 10k)
        onk = (att * (seqs.unsqueeze(1) == k)).sum(-1)  # (B, 10q)
        m = onk[qmask].mean()
        if best is None or m > best[1]:
            best = (k, m.item())
    print(f"  L0H{h}: attends to '{chr(97+best[0])}' with total weight {best[1]:.2f} when in prefix")
