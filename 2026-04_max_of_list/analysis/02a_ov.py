import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load("a")
E, P, U, L = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight, m.layers[0]
D = 16
def ov(h):
    return L.W_O.weight[:, h * D:(h + 1) * D] @ L.heads[h].W_V.weight

print("OV circuit per head: logit (over answers 0..9) produced by fully attending to number n at a number position (avg over positions)")
for h in range(4):
    M = torch.zeros(10, 10)
    for n in range(10):
        v = torch.stack([ov(h) @ (E[n] + P[p]) for p in [1, 3, 5, 7, 9]]).mean(0)
        M[n] = (U[:10] @ v)
    diag = M.diag().mean().item(); off = (M.sum() - M.diag().sum()).item() / 90
    print(f"\n  H{h}: mean diagonal {diag:6.2f}, mean off-diagonal {off:6.2f}   (rows = attended n, cols = logit k)")
    for n in range(10):
        print(f"     n={n}: " + " ".join(f"{v:6.1f}" for v in M[n].tolist()))
    self_v = ov(h) @ (E[ANS] + P[10])
    print(f"     attending to ANS itself gives logits: " + " ".join(f"{v:6.1f}" for v in (U[:10] @ self_v).tolist()))
print("\nDirect path E[ANS]+P[10] -> logits 0..9:", (U[:10] @ (E[ANS] + P[10])).numpy().round(1))

# ablations
rng = np.random.default_rng(1)
lists = rng.integers(0, 10, size=(4000, 5))
x = torch.tensor([tok1(l.tolist()) for l in lists]); true = torch.tensor(lists.max(1))
def run(x, zero_heads=()):
    b, s = x.shape
    h = E[x] + P[torch.arange(s)]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    outs = []
    for hi, head in enumerate(L.heads):
        o, _ = head(h, mask)
        if hi in zero_heads: o = torch.zeros_like(o)
        outs.append(o)
    h = h + L.W_O(torch.cat(outs, -1))
    return h[:, -1] @ U.T
print("\nZero-ablate heads at all positions -> accuracy on max:")
for zh in [(), (0,), (1,), (2,), (3,), (0, 1, 2), (0, 2), (1, 2), (0, 1)]:
    lg = run(x, zh)
    acc = (lg[:, :10].argmax(-1) == true).float().mean().item()
    # accuracy split by true max value
    by = {v: (lg[:, :10].argmax(-1) == true)[true == v].float().mean().item() for v in range(10)}
    print(f"  zero {str(zh):12s}: acc = {acc:.3f}   by max value: " + " ".join(f"{v}:{by[v]:.2f}" for v in range(10)))
print("\nkeep only head 3 (zero 0,1,2):", (run(x, (0, 1, 2))[:, :10].argmax(-1) == true).float().mean().item())
