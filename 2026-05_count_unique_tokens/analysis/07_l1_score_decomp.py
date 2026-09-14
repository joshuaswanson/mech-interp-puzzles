import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight

def layer0_out(x):
    b, s = x.shape
    h = E[x]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    layer = model.layers[0]
    outs = [head(h, mask)[0] for head in layer.heads]
    return layer.W_O(torch.cat(outs, -1))

rng = np.random.default_rng(4)
x, counts = random_batch(rng, 300)
B = len(counts)
L0 = layer0_out(x)         # (B, 12, 32)
Emb = E[x]                 # (B, 12, 32)
seqs = x[:, 1:-1]
first = torch.zeros(B, SEQ_LEN, dtype=torch.bool)
prefix_count = torch.zeros(B, SEQ_LEN)
for b in range(B):
    seen = {}
    for i, t in enumerate(seqs[b].tolist()):
        first[b, i] = t not in seen
        seen[t] = seen.get(t, 0) + 1
        prefix_count[b, i] = seen[t]

for hi, head in enumerate(model.layers[1].heads):
    WQ, WK = head.W_Q.weight, head.W_K.weight
    qE = Emb[:, -1] @ WQ.T
    qL = L0[:, -1] @ WQ.T
    kE = Emb @ WK.T
    kL = L0 @ WK.T
    sc = (1 / D_HEAD ** 0.5)
    terms = {
        "qE.kE": torch.einsum("bd,bjd->bj", qE, kE) * sc,
        "qE.kL0": torch.einsum("bd,bjd->bj", qE, kL) * sc,
        "qL0.kE": torch.einsum("bd,bjd->bj", qL, kE) * sc,
        "qL0.kL0": torch.einsum("bd,bjd->bj", qL, kL) * sc,
    }
    total = sum(terms.values())
    print(f"\n===== L1H{hi}: ANS-row score decomposition =====")
    print("Score at BOS key (mean per count):")
    for name, t in list(terms.items()) + [("total", total)]:
        print(f"  {name:8s} " + " ".join(f"{t[counts == c, 0].mean():7.2f}" for c in range(1, 11)))
    print("Score at ANS key (mean per count):")
    for name, t in list(terms.items()) + [("total", total)]:
        print(f"  {name:8s} " + " ".join(f"{t[counts == c, -1].mean():7.2f}" for c in range(1, 11)))
    print("Score at symbol keys: mean over FIRST occurrences vs REPEATS, per symbol identity")
    for name, t in list(terms.items()) + [("total", total)]:
        ts = t[:, 1:-1]
        fs = " ".join(f"{ts[(seqs == s) & first].mean():6.2f}" for s in range(10))
        rs = " ".join(f"{ts[(seqs == s) & ~first].mean():6.2f}" for s in range(10))
        print(f"  {name:8s} first: {fs}")
        print(f"  {'':8s} rept : {rs}")
    print("Total score at symbol keys by prefix_count of that symbol (1=first occ):")
    ts = total[:, 1:-1]
    print("   " + " ".join(f"k={k}:{ts[prefix_count == k].mean():6.2f}" for k in range(1, 8)))
