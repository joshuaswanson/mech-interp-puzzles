import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load_model()
E, P, U, L = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight, m.layers[0]
ORDER = "dahgbkefionplcjm"

# one fixed ordering per (X, z), all 7280 problems
prompts = []
for X in itertools.combinations(range(16), 4):
    for z in X:
        Y = [s for s in X if s != z]
        prompts.append((encode(X, Y), z))
x = torch.tensor([p for p, _ in prompts]); z = torch.tensor([zz for _, zz in prompts])
logits, attn = m(x)
pred = logits[:, -1, :16].argmax(-1)
print("accuracy (one ordering per problem):", (pred == z).float().mean().item())

# head scalar outputs at the final position
r = E[x] + P[torch.arange(10)]
s = []
for h in range(2):
    hd = L.heads[h]
    v = (r @ hd.W_V.weight.T)[:, :, 0]         # (B, 10)
    s.append((attn[0][:, h, -1] * v).sum(-1))  # (B,)
s = torch.stack(s, 1)                          # (B, 2)
a0 = L.W_O.weight[:, 0]; a1 = L.W_O.weight[:, 1]
r9 = E[SEP] + P[9]
final = r9 + s[:, :1] * a0 + s[:, 1:] * a1
assert torch.allclose(final @ U.T, logits[:, -1], atol=1e-3)
ang = torch.rad2deg(torch.atan2(final[:, 1], final[:, 0]))
uang = torch.rad2deg(torch.atan2(U[:16, 1], U[:16, 0]))

print("\nFinal residual per missing symbol z (in embedding order):")
print(f"{'z':>2s} {'n':>4s} {'s0 mean':>8s} {'s0 std':>7s} {'s1 mean':>8s} {'s1 std':>7s} | {'angle mean':>10s} {'angle min':>9s} {'angle max':>9s} | {'|r| mean':>8s} | {'U angle':>7s}")
for ch in ORDER:
    k = ord(ch) - 97
    sel = z == k
    print(f"{ch:>2s} {int(sel.sum()):4d} {s[sel,0].mean():8.2f} {s[sel,0].std():7.2f} {s[sel,1].mean():8.2f} {s[sel,1].std():7.2f} | {ang[sel].mean():10.1f} {ang[sel].min():9.1f} {ang[sel].max():9.1f} | {final[sel].norm(dim=1).mean():8.2f} | {uang[k]:7.1f}")

# Is (s0, s1) essentially one-dimensional?  Fit s1 as a function of s0.
A = torch.stack([s[:, 0], torch.ones(len(s))], 1)
w = torch.linalg.lstsq(A, s[:, 1:]).solution.squeeze()
res = s[:, 1] - (A @ w)
print(f"\ns1 ~ {w[0]:.3f} * s0 + {w[1]:.3f}:  R^2 = {1 - res.var() / s[:, 1].var():.4f}")
print("correlation(s0, s1) =", float(torch.corrcoef(s.T)[0, 1]))
