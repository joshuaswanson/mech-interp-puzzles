import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight
U = model.unembed.weight[COUNT_BASE:]

def forward(x, l1_attn_override=None):
    b, s = x.shape
    h = E[x]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    attn_rows = {}
    for L, layer in enumerate(model.layers):
        outs = []
        for hi, head in enumerate(layer.heads):
            o, a = head(h, mask)
            if L == 1:
                attn_rows[hi] = a[:, -1]
            if L == 1 and l1_attn_override and hi in l1_attn_override:
                v = head.W_V(h)
                o = o.clone()
                o[:, -1] = torch.einsum("bj,bjd->bd", l1_attn_override[hi], v)
            outs.append(o)
        h = h + layer.W_O(torch.cat(outs, -1))
    return (h[:, -1] @ U.T).argmax(-1) + 1, attn_rows

rng = np.random.default_rng(11)
def make_pairs(target, n=400):
    xs, xps = [], []
    while len(xs) < n:
        c = int(rng.integers(2, 11))
        s = random_sequence(rng, c)
        if target not in s:
            continue
        others = [t for t in set(s) if t != target]
        repl = others[int(rng.integers(len(others)))]
        xs.append([BOS] + s + [ANS]); xps.append([BOS] + [repl if t == target else t for t in s] + [ANS])
    return torch.tensor(xs), torch.tensor(xps)

print("Override layer-1 head h's ANS attention row on S with the row it produces on S' (S with X removed).")
print("Only the attention pattern changes; values are from S.\n")
for hi, X in [(0, 1), (1, 5), (2, 0), (2, 3), (2, 6), (2, 7), (3, 2), (3, 4), (3, 8), (3, 9)]:
    x, xp = make_pairs(X)
    Utrue = torch.tensor([len(set(r[1:-1].tolist())) for r in x])
    _, rows_p = forward(xp)
    pred, _ = forward(x, l1_attn_override={hi: rows_p[hi]})
    print(f"  L1H{hi}, X='{chr(97+X)}':  pred=U-1: {(pred==Utrue-1).float().mean():.3f}   pred=U: {(pred==Utrue).float().mean():.3f}")

print("\nSame, but override L1H2's attention with the pattern from S' where X in {a,d,g,h} removed: values at the")
print("attended position in S still contain the X signal from layer 0, so the count should NOT drop if the new")
print("position's prefix is still complete. (Tests that L1H2 counts via VALUES, and the pattern only picks a position.)")
# Already shown above for hi=2. Now the converse: keep L1H2 pattern, but patch VALUES (layer-0 residual at symbol positions) from S'.
def forward_patch_resid(x, x_src, positions="symbols"):
    b, s = x.shape
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    def l0(xx):
        hh = E[xx]
        outs = [head(hh, mask)[0] for head in model.layers[0].heads]
        return hh + model.layers[0].W_O(torch.cat(outs, -1))
    h = l0(x); hs = l0(x_src)
    hp = h.clone()
    if positions == "symbols":
        hp[:, 1:-1] = hs[:, 1:-1]
    layer = model.layers[1]
    outs = []
    for hi, head in enumerate(layer.heads):
        q = head.W_Q(h); k = head.W_K(h); v = head.W_V(hp if hi == 2 else h)
        sc = torch.einsum("bid,bjd->bij", q, k) / D_HEAD ** 0.5
        sc = sc.masked_fill(mask == 0, float("-inf"))
        a = torch.softmax(sc, -1)
        outs.append(torch.einsum("bij,bjd->bid", a, v))
    h = h + layer.W_O(torch.cat(outs, -1))
    return (h[:, -1] @ U.T).argmax(-1) + 1

print("\nPatch only the VALUE inputs of L1H2 (layer-0 residual at symbol positions) from S' (X removed), keep pattern from S:")
for X in [0, 3, 6, 7, 2, 8]:
    x, xp = make_pairs(X)
    Utrue = torch.tensor([len(set(r[1:-1].tolist())) for r in x])
    pred = forward_patch_resid(x, xp)
    print(f"  X='{chr(97+X)}':  pred=U-1: {(pred==Utrue-1).float().mean():.3f}   pred=U: {(pred==Utrue).float().mean():.3f}")
