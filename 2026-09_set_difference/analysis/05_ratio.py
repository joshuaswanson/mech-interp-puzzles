import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load_model()
E, P, U, L = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight, m.layers[0]
ORDER = "dahgbkefionplcjm"
r9 = E[SEP] + P[9]

def head_parts(toks, h):
    hd = L.heads[h]
    r = E[torch.tensor(toks)] + P[torch.arange(10)]
    q = float(hd.W_Q.weight[0] @ r9)
    e = torch.exp(q * (r @ hd.W_K.weight[0]))
    v = r @ hd.W_V.weight[0]
    return e, v

prompts = []
for X in itertools.combinations(range(16), 4):
    for z in X:
        Y = [s for s in X if s != z]
        prompts.append((encode(X, Y), z))

print("Same missing symbol, different companions: the two head outputs scale together.")
for zc in ["d", "i", "m"]:
    zt = ord(zc) - 97
    print(f"\nz = {zc}")
    print(f"{'prompt':32s} {'N0':>7s} {'Z0':>6s} {'s0':>7s} | {'N1':>7s} {'Z1':>6s} {'s1':>7s} | {'s1/s0':>6s} {'angle':>6s}")
    shown = 0
    for toks, z in prompts:
        if z != zt or shown >= 7: continue
        shown += 1
        e0, v0 = head_parts(toks, 0); e1, v1 = head_parts(toks, 1)
        s0 = float((e0 * v0).sum() / e0.sum()); s1 = float((e1 * v1).sum() / e1.sum())
        fin = r9 + s0 * L.W_O.weight[:, 0] + s1 * L.W_O.weight[:, 1]
        ang = float(torch.rad2deg(torch.atan2(fin[1], fin[0])))
        print(f"{' '.join(label(t) for t in toks):32s} {float((e0*v0).sum()):7.1f} {float(e0.sum()):6.2f} {s0:7.2f} | {float((e1*v1).sum()):7.1f} {float(e1.sum()):6.2f} {s1:7.2f} | {s1/s0:6.2f} {ang:6.1f}")

# Ratio s1/s0 across all prompts per z
x = torch.tensor([p for p, _ in prompts]); z = torch.tensor([zz for _, zz in prompts])
_, attn = m(x)
r = E[x] + P[torch.arange(10)]
s = torch.stack([(attn[0][:, h, -1] * (r @ L.heads[h].W_V.weight[0])).sum(-1) for h in range(2)], 1)
print("\nratio s1/s0 per z over all 7280 problems (one ordering each):")
for ch in ORDER:
    k = ord(ch) - 97; sel = z == k
    ratio = s[sel, 1] / s[sel, 0]
    print(f"  {ch}: mean {ratio.mean():8.3f}  std {ratio.std():7.3f}   (s0 range {s[sel,0].min():6.2f} .. {s[sel,0].max():6.2f})")
