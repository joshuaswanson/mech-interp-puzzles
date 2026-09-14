import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load("a")
E, P, U, L = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight, m.layers[0]
D = 16
def ov(h): return L.W_O.weight[:, h * D:(h + 1) * D] @ L.heads[h].W_V.weight

print("Rank of each head's OV path into the 10 answer logits (singular values of U[:10] @ W_OV restricted to the number embeddings)")
for h in range(4):
    M = torch.stack([U[:10] @ (ov(h) @ (E[n] + P[5])) for n in range(10)])   # (10 attended n, 10 logits)
    sv = torch.linalg.svdvals(M)
    print(f"  H{h}: {sv[:4].numpy().round(1)}   -> fraction in first component {float(sv[0]**2 / (sv**2).sum()):.3f}")

print("\nToken embeddings of the digits: singular values of E[0..9] (centered)")
Ec = E[:10] - E[:10].mean(0)
print("  ", torch.linalg.svdvals(Ec)[:6].numpy().round(2))

# Per-max-value logit decomposition on random inputs
rng = np.random.default_rng(2)
lists = rng.integers(0, 10, size=(20000, 5))
x = torch.tensor([tok1(l.tolist()) for l in lists]); true = torch.tensor(lists.max(1))
b, s = x.shape
h = E[x] + P[torch.arange(s)]
mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
contrib = {"direct": h[:, -1] @ U[:10].T}
for hi, head in enumerate(L.heads):
    o, _ = head(h, mask)
    contrib[f"H{hi}"] = (o[:, -1] @ L.W_O.weight[:, hi * D:(hi + 1) * D].T) @ U[:10].T
total = sum(contrib.values())
assert (total.argmax(-1) == true).float().mean() == 1.0
print("\nFor each true max M: mean logit vector (over 0..9) from each component, and the margin of class M over the runner-up in the total")
for M in range(2, 10):
    sel = true == M
    if sel.sum() == 0: continue
    print(f"\n  M = {M}  (n = {int(sel.sum())})")
    for name, c in contrib.items():
        row = c[sel].mean(0)
        print(f"    {name:7s}: " + " ".join(f"{v:7.1f}" for v in row.tolist()) + f"   argmax {int(row.argmax())}")
    row = total[sel].mean(0)
    top2 = total[sel].topk(2, dim=-1).values
    print(f"    {'total':7s}: " + " ".join(f"{v:7.1f}" for v in row.tolist()) + f"   argmax {int(row.argmax())}   min margin {(top2[:,0]-top2[:,1]).min():.1f}")
