import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load("b")
E, P, U = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight
L0, L1 = m.layers
D = 16
TENS = [1, 4, 7, 10, 13]; ONES = [2, 5, 8, 11, 14]

print("A. L0H1 from ANS (pos 15): score to tens-digit token d at tens positions (mean over positions), and to ones positions")
hd = L0.heads[1]; q = hd.W_Q(E[ANS] + P[15])
print("   tens digit d: " + " ".join(f"{d}:{float(torch.stack([hd.W_K(E[d]+P[p]) for p in TENS]).mean(0) @ q / 4):6.2f}" for d in range(10)))
print("   same token at ones positions: " + " ".join(f"{d}:{float(torch.stack([hd.W_K(E[d]+P[p]) for p in ONES]).mean(0) @ q / 4):6.2f}" for d in range(10)))
print(f"   ANS self: {float(hd.W_K(E[ANS]+P[15]) @ q / 4):.2f}   BOS: {float(hd.W_K(E[BOS]+P[0]) @ q / 4):.2f}")

print("\nB. L0H3 from a ones position p to earlier positions: score by relative offset (token-averaged), i.e. is it a previous-token head?")
hd = L0.heads[3]
for p in [2, 8, 14]:
    row = []
    for j in range(0, p + 1):
        toks = [BOS] if j == 0 else (list(range(10)) if j % 3 != 0 else [SEP])
        s = np.mean([float(hd.W_K(E[t] + P[j]) @ hd.W_Q(E[o] + P[p]) / 4) for t in toks for o in range(10)])
        row.append(s)
    print(f"   query ones pos {p:2d}: " + " ".join(f"{j}:{s:5.1f}" for j, s in enumerate(row)))

print("\nC. What L0H3 writes at a ones position when it copies tens digit t: read through L1H0's key and the pos-16 query of token T")
# residual at ones position after layer 0 ~ E[o] + P[p] + W_OV(L0H3)(E[t] + P[p-1]) + other heads. Build the match matrix score[T, t].
def ov(layer, h): return layer.W_O.weight[:, h * D:(h + 1) * D] @ layer.heads[h].W_V.weight
for h1 in [0, 3]:
    hd1 = L1.heads[h1]
    print(f"  L1H{h1}: score contribution (query token T at pos 16) x (key = L0H3 copy of tens digit t at a ones position), avg over positions")
    M = torch.zeros(10, 10)
    for T in range(10):
        q = hd1.W_Q(E[T] + P[16])
        for t in range(10):
            k = torch.stack([hd1.W_K(ov(L0, 3) @ (E[t] + P[p - 1])) for p in ONES]).mean(0)
            M[T, t] = k @ q / 4
    print("      rows = T (query), cols = t (tens digit stored at the key)")
    for T in range(10):
        print(f"      T={T}: " + " ".join(f"{v:6.1f}" for v in M[T].tolist()))
    print(f"  L1H{h1}: score contribution of the ones digit o itself at a ones position (E[o] + P[p]), avg over T and positions")
    row = [np.mean([float(hd1.W_K(E[o] + P[p]) @ hd1.W_Q(E[T] + P[16]) / 4) for p in ONES for T in range(10)]) for o in range(10)]
    print("      " + " ".join(f"{o}:{v:6.2f}" for o, v in enumerate(row)))

print("\nD. Ablations (3000 random lists)")
rng = np.random.default_rng(3)
lists = rng.integers(0, 100, size=(3000, 5))
x = torch.tensor([tok2(l.tolist()) for l in lists]); true = torch.tensor(lists.max(1)); T, O = true // 10, true % 10
x2 = torch.cat([x, T[:, None]], 1)
def run(x, zero=()):
    b, s = x.shape
    h = E[x] + P[torch.arange(s)]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    for L, layer in enumerate(m.layers):
        outs = []
        for hi, head in enumerate(layer.heads):
            o, _ = head(h, mask)
            if (L, hi) in zero: o = torch.zeros_like(o)
            outs.append(o)
        h = h + layer.W_O(torch.cat(outs, -1))
    return h[:, -1] @ U.T
for name, zero in [("none", ()), ("L0H1", ((0, 1),)), ("L0H3", ((0, 3),)), ("L1H0", ((1, 0),)), ("L1H3", ((1, 3),)), ("L1H0+L1H3", ((1, 0), (1, 3))), ("all of layer 1", tuple((1, h) for h in range(4))), ("L0H0+L0H2", ((0, 0), (0, 2)))]:
    accT = (run(x, zero)[:, :10].argmax(-1) == T).float().mean().item()
    accO = (run(x2, zero)[:, :10].argmax(-1) == O).float().mean().item()
    print(f"   zero {name:14s}: tens acc {accT:.3f}   ones acc {accO:.3f}")
