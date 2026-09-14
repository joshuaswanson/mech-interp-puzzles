import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load_model()
E, P, U, L = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight, m.layers[0]
ORDER = "dahgbkefionplcjm"
r9 = E[SEP] + P[9]
PX, PY = P[1], P[6]

print("1. Pair cancellation: e_X v_X + e_Y v_Y for each symbol (should be constant per head)")
for h in range(2):
    hd = L.heads[h]
    q = float(hd.W_Q.weight[0] @ r9); wk, wv = hd.W_K.weight[0], hd.W_V.weight[0]
    vals, lone = [], []
    for ch in ORDER:
        t = ord(ch) - 97
        eX = float(torch.exp(q * (wk @ (E[t] + PX)))); vX = float(wv @ (E[t] + PX))
        eY = float(torch.exp(q * (wk @ (E[t] + PY)))); vY = float(wv @ (E[t] + PY))
        vals.append(eX * vX + eY * vY); lone.append(eX * vX)
    rho = float(torch.exp(q * (wk @ (PY - PX)))); dv = float(wv @ (PY - PX))
    print(f"  head {h}: pair contribution per symbol: " + " ".join(f"{c}={v:6.1f}" for c, v in zip(ORDER, vals)))
    print(f"          mean {np.mean(vals):.1f}, std {np.std(vals):.2f};   lone-z contribution g(z) ranges {min(lone):.1f} .. {max(lone):.1f}")
    print(f"          e_Y/e_X = {rho:.3f} (same for every symbol), v_Y - v_X = {dv:.1f} (same for every symbol)")
    # why: v_X must be affine in 1/e_X  ->  check linearity of v_X vs exp(-score_X)
    xs = np.array([1 / float(torch.exp(q * (wk @ (E[ord(c)-97] + PX)))) for c in ORDER])
    ys = np.array([float(wv @ (E[ord(c)-97] + PX)) for c in ORDER])
    A = np.stack([xs, np.ones_like(xs)], 1); coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    r2 = 1 - ((A @ coef - ys) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    print(f"          v_X(t) vs 1/e_X(t): linear fit v = {coef[0]:.2f}/e_X + {coef[1]:.2f}, R^2 = {r2:.4f}  (the embedding curve)")

print("\n2. Normaliser ratio Z0/Z1 over all problems, and the angle bands vs decision boundaries")
prompts = []
for X in itertools.combinations(range(16), 4):
    for z in X:
        Y = [s for s in X if s != z]
        prompts.append((encode(X, Y), z))
x = torch.tensor([p for p, _ in prompts]); z = torch.tensor([zz for _, zz in prompts])
r = E[x] + P[torch.arange(10)]
Zs, Ns = [], []
for h in range(2):
    hd = L.heads[h]
    q = float(hd.W_Q.weight[0] @ r9)
    e = torch.exp(q * (r @ hd.W_K.weight[0])); v = r @ hd.W_V.weight[0]
    Zs.append(e.sum(-1)); Ns.append((e * v).sum(-1))
Z0, Z1 = Zs; N0, N1 = Ns
print(f"  Z0/Z1: min {float((Z0/Z1).min()):.3f}, mean {float((Z0/Z1).mean()):.3f}, max {float((Z0/Z1).max()):.3f}")
for ch in ORDER:
    k = ord(ch) - 97; sel = z == k
    print(f"  z={ch}: N0 = {N0[sel].mean():7.1f} ± {N0[sel].std():4.1f}   N1 = {N1[sel].mean():7.1f} ± {N1[sel].std():4.1f}")

print("\n3. Readout: a fan of unembedding vectors. For each direction angle, which symbol has the largest logit?")
angles = torch.arange(-40, 130, 0.5)
dirs = torch.stack([torch.cos(torch.deg2rad(angles)), torch.sin(torch.deg2rad(angles))], 1)
win = (dirs @ U[:16].T).argmax(-1)
segments = []
start = 0
for i in range(1, len(angles)):
    if win[i] != win[i - 1]:
        segments.append((label(int(win[start])), float(angles[start]), float(angles[i - 1]))); start = i
segments.append((label(int(win[start])), float(angles[start]), float(angles[-1])))
print("  winner by angle: " + ", ".join(f"{s} [{a:.0f}°..{b:.0f}°]" for s, a, b in segments))
final = r9 + N0[:, None] / Z0[:, None] * L.W_O.weight[:, 0] + N1[:, None] / Z1[:, None] * L.W_O.weight[:, 1]
ang = torch.rad2deg(torch.atan2(final[:, 1], final[:, 0]))
print("  observed band per z: " + ", ".join(f"{ch} [{ang[z == ord(ch)-97].min():.1f}..{ang[z == ord(ch)-97].max():.1f}]" for ch in ORDER))
logits = final @ U[:16].T
top2 = logits.topk(2, dim=-1).values
print(f"  min logit margin over all problems: {(top2[:, 0] - top2[:, 1]).min():.2f}")
print(f"  direct path |r9| = {r9.norm():.2f} vs mean |final| = {final.norm(dim=1).mean():.1f}")
