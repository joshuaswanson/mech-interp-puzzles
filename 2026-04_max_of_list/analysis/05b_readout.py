import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load("b")
E, P, U = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight
L0, L1 = m.layers
D = 16

def decompose(x):
    """Per-component logit contributions at the last position: direct, each L0 head, each L1 head."""
    b, s = x.shape
    h = E[x] + P[torch.arange(s)]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    comps = {"direct": h[:, -1] @ U[:10].T}
    for L, layer in enumerate(m.layers):
        outs = []
        for hi, head in enumerate(layer.heads):
            o, _ = head(h, mask); outs.append(o)
            comps[f"L{L}H{hi}"] = (o[:, -1] @ layer.W_O.weight[:, hi * D:(hi + 1) * D].T) @ U[:10].T
        h = h + layer.W_O(torch.cat(outs, -1))
    return comps

rng = np.random.default_rng(6)
lists = rng.integers(0, 100, size=(6000, 5))
x = torch.tensor([tok2(l.tolist()) for l in lists]); true = torch.tensor(lists.max(1)); T, O = true // 10, true % 10
x2 = torch.cat([x, T[:, None]], 1)

for step, xx, target, name in [("tens", x, T, "T"), ("ones", x2, O, "O")]:
    comps = decompose(xx)
    total = sum(comps.values())
    print(f"\n=== {step} step: accuracy {(total.argmax(-1) == target).float().mean():.3f}. Mean logit vector per component for each true {name} (rows), argmax of each, and which components are needed ===")
    for v in range(10):
        sel = target == v
        if sel.sum() < 5: continue
        line = f"  {name}={v}: "
        for k, c in comps.items():
            line += f"{k}:{int(c[sel].mean(0).argmax())} "
        top2 = total[sel].topk(2, dim=-1).values
        print(line + f"| total argmax {int(total[sel].mean(0).argmax())}, min margin {(top2[:,0]-top2[:,1]).min():.1f}")
    print("  (numbers are the argmax over 0..9 of each component's mean logit vector; a component that tracks the answer shows the row value)")
    # rank of the answer-carrying head's path
    for (L, hi) in ([(0, 1)] if step == "tens" else [(1, 0), (1, 3)]):
        layer = m.layers[L]
        ov = layer.W_O.weight[:, hi * D:(hi + 1) * D] @ layer.heads[hi].W_V.weight
        M = torch.stack([U[:10] @ (ov @ (E[d] + P[5])) for d in range(10)])
        sv = torch.linalg.svdvals(M)
        print(f"  L{L}H{hi} OV path digit->logits: singular values {sv[:3].numpy().round(1)}, diagonal mean {M.diag().mean():.1f}, off-diag mean {(M.sum()-M.diag().sum())/90:.1f}")
