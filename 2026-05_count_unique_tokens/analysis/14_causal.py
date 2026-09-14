import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight
U = model.unembed.weight[COUNT_BASE:]

def forward(x, patch=None, l1_attn_override=None):
    """patch: dict {(layer0_head, 'symbols'|'ans'): tensor (B,12,8)} head outputs (pre-W_O) to substitute.
       l1_attn_override: dict {head: (B,12) attention row for ANS query}"""
    b, s = x.shape
    h = E[x]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    for L, layer in enumerate(model.layers):
        outs = []
        for hi, head in enumerate(layer.heads):
            if L == 1 and l1_attn_override and hi in l1_attn_override:
                v = head.W_V(h)
                a = l1_attn_override[hi]
                o = torch.einsum("bj,bjd->bd", a, v)
                oo = head(h, mask)[0].clone()
                oo[:, -1] = o
                o = oo
            else:
                o, _ = head(h, mask)
            if L == 0 and patch:
                if (hi, "symbols") in patch:
                    o = o.clone(); o[:, 1:-1] = patch[(hi, "symbols")][:, 1:-1]
                if (hi, "ans") in patch:
                    o = o.clone(); o[:, -1] = patch[(hi, "ans")][:, -1]
            outs.append(o)
        h = h + layer.W_O(torch.cat(outs, -1))
    return (h[:, -1] @ U.T).argmax(-1) + 1

def head_outputs_l0(x):
    b, s = x.shape
    h = E[x]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    return [head(h, mask)[0] for head in model.layers[0].heads]

rng = np.random.default_rng(10)

def make_pairs(target, n=400):
    """Sequences S containing `target` and >=2 unique symbols; S' = S with target replaced by another present symbol."""
    xs, xps = [], []
    while len(xs) < n:
        c = int(rng.integers(2, 11))
        s = random_sequence(rng, c)
        if target not in s:
            continue
        others = [t for t in set(s) if t != target]
        repl = others[int(rng.integers(len(others)))]
        sp = [repl if t == target else t for t in s]
        xs.append([BOS] + s + [ANS]); xps.append([BOS] + sp + [ANS])
    return torch.tensor(xs), torch.tensor(xps)

print("Patch a single layer-0 head's output (from S' = S with symbol X replaced) into the run on S.")
print("If that head is the X-detector feeding the readout, prediction on S should become U-1.\n")
print(f"{'head':6s} {'where':8s} {'X':>2s}  {'base acc':>8s}  {'pred=U-1':>9s}  {'pred=U':>7s}")
detectors = {("symbols", 0): 0, ("symbols", 1): 7, ("symbols", 2): 3, ("symbols", 3): 6,
             ("ans", 0): 8, ("ans", 1): 9, ("ans", 2): 2, ("ans", 3): 4}
for (where, hi), X in detectors.items():
    x, xp = make_pairs(X)
    Utrue = torch.tensor([len(set(r[1:-1].tolist())) for r in x])
    base = forward(x)
    ho_p = head_outputs_l0(xp)
    pred = forward(x, patch={(hi, where): ho_p[hi]})
    print(f"L0H{hi}   {where:8s} {chr(97+X):>2s}  {(base==Utrue).float().mean():8.3f}  {(pred==Utrue-1).float().mean():9.3f}  {(pred==Utrue).float().mean():7.3f}")

print("\nControl: patch the SAME head from S' where a NON-target symbol was replaced (should keep pred=U):")
print(f"{'head':6s} {'where':8s} {'X':>2s}  {'pred=U-1':>9s}  {'pred=U':>7s}")
for (where, hi), X in detectors.items():
    Y = (X + 1) % 10
    while Y in detectors.values() and (where, hi) in detectors and Y == detectors[(where, hi)]:
        Y = (Y + 1) % 10
    # choose a control symbol not detected by this head at this position
    Y = {0: 2, 1: 9, 2: 6, 3: 8}[hi] if where == "symbols" else {0: 4, 1: 7, 2: 8, 3: 8}[hi]
    x, xp = make_pairs(Y)
    Utrue = torch.tensor([len(set(r[1:-1].tolist())) for r in x])
    ho_p = head_outputs_l0(xp)
    pred = forward(x, patch={(hi, where): ho_p[hi]})
    print(f"L0H{hi}   {where:8s} {chr(97+Y):>2s}  {(pred==Utrue-1).float().mean():9.3f}  {(pred==Utrue).float().mean():7.3f}")

print("\nLayer-1 attention overrides at ANS:")
for hi, X, name in [(0, 1, "b"), (1, 5, "f")]:
    x, _ = make_pairs(X)
    Utrue = torch.tensor([len(set(r[1:-1].tolist())) for r in x])
    a = torch.zeros(len(x), 12); a[:, 0] = 1.0
    pred = forward(x, l1_attn_override={hi: a})
    print(f"  L1H{hi} forced to attend BOS on sequences containing '{name}': pred=U-1: {(pred==Utrue-1).float().mean():.3f}  pred=U: {(pred==Utrue).float().mean():.3f}")
    # sequences WITHOUT X: force attention to a position that holds X? can't. Instead: without X, force BOS should keep U.
    xs = []
    while len(xs) < 400:
        s = random_sequence(rng, int(rng.integers(1, 10)))
        if X not in s: xs.append([BOS] + s + [ANS])
    xs = torch.tensor(xs)
    Utrue2 = torch.tensor([len(set(r[1:-1].tolist())) for r in xs])
    pred2 = forward(xs, l1_attn_override={hi: a})
    print(f"  L1H{hi} forced to attend BOS on sequences WITHOUT '{name}': pred=U: {(pred2==Utrue2).float().mean():.3f}")
