import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load("b")
E, P, U = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight
L0, L1 = m.layers
D = 16
ONES = [2, 5, 8, 11, 14]

def l0_parts(x):
    b, s = x.shape
    h = E[x] + P[torch.arange(s)]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    parts, outs, attns = {"tok+pos": h}, [], {}
    for hi, head in enumerate(L0.heads):
        o, a = head(h, mask); outs.append(o); attns[hi] = a
        parts[f"L0H{hi}"] = o @ L0.W_O.weight[:, hi * D:(hi + 1) * D].T
    return h + L0.W_O(torch.cat(outs, -1)), parts, attns

print("Fixed list [8o, 12, 45, 60, 33] with the ones digit o of the first number varied. Query = output position with token 8.")
print("L1H0 score at the first ones position (pos 2), split by which part of that position's residual carries it:")
print(f"{'o':>2s} {'tok+pos':>8s} {'L0H0':>7s} {'L0H1':>7s} {'L0H2':>7s} {'L0H3':>7s} {'total':>8s} | {'L0H3 attn on own tens':>22s} {'|L0H3 out|':>10s}")
for o in range(10):
    nums = [80 + o, 12, 45, 60, 33]
    x = torch.tensor([tok2(nums) + [8]])
    r1, parts, attns = l0_parts(x)
    hd = L1.heads[0]; q = hd.W_Q(r1[0, 16])
    sc = {k: float(hd.W_K(v[0, 2]) @ q / 4) for k, v in parts.items()}
    tot = float(hd.W_K(r1[0, 2]) @ q / 4)
    print(f"{o:2d} {sc['tok+pos']:8.2f} {sc['L0H0']:7.2f} {sc['L0H1']:7.2f} {sc['L0H2']:7.2f} {sc['L0H3']:7.2f} {tot:8.2f} | {float(attns[3][0, 2, 1]):22.3f} {float(parts['L0H3'][0, 2].norm()):10.2f}")

print("\nSame, but now the L0H3 value at the tens position is what gets copied. Does the L1H0 key direction read 'tens digit' or 'ones-modulated copy strength'?")
print("Score from the pure copy W_OV^{L0H3}(E[t] + P[1]) alone, for t = 0..9 (query token 8 at pos 16):")
ov3 = L0.W_O.weight[:, 3 * D:4 * D] @ L0.heads[3].W_V.weight
hd = L1.heads[0]
x = torch.tensor([tok2([83, 12, 45, 60, 33]) + [8]]); r1, _, _ = l0_parts(x); q = hd.W_Q(r1[0, 16])
print("   " + " ".join(f"t={t}:{float(hd.W_K(ov3 @ (E[t] + P[1])) @ q / 4):6.1f}" for t in range(10)))
print("So one unit of attention on tens digit 8 is worth", round(float(hd.W_K(ov3 @ (E[8] + P[1])) @ q / 4), 1), "and the ones digit changes the attention weight, which scales this.")

print("\nL0H3 attention from a ones position to its own tens position, as a function of the ones digit (avg over slots and tens digits):")
rng = np.random.default_rng(5)
acc = np.zeros(10); cnt = np.zeros(10)
for _ in range(400):
    nums = rng.integers(0, 100, size=5).tolist()
    x = torch.tensor([tok2(nums)]); _, _, attns = l0_parts(x)
    for i, p in enumerate(ONES):
        o = nums[i] % 10; acc[o] += float(attns[3][0, p, p - 1]); cnt[o] += 1
print("   " + " ".join(f"o={o}:{acc[o]/cnt[o]:.3f}" for o in range(10)))
