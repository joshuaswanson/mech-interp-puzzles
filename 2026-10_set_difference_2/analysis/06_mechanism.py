import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()
E, U = model.tok_embed.weight, model.unembed.weight
Uf = U * model.ln_final.weight
b0, b1 = model.blocks[0], model.blocks[1]

def run(tokens, l0_override=None, l1_override=None):
    x = torch.tensor([tokens]); n = len(tokens)
    rope = mod.rope_cos_sin(n, 64, 10000.0, x.device)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    def attend(blk, r, override):
        h = blk.ln_attn(r)
        q, k, v = blk.attn.W_Q(h), blk.attn.W_K(h), blk.attn.W_V(h)
        cos, sin = rope
        q = q * cos + mod.rotate_half(q) * sin
        k = k * cos + mod.rotate_half(k) * sin
        sc = ((q @ k.transpose(-1, -2)) / 8.0).masked_fill(~mask, float("-inf"))
        p = sc.softmax(-1)
        if override:
            p = p.clone()
            for pos, row in override.items(): p[0, pos] = row
        return blk.attn.W_O(p @ v), p
    r0 = E[x]
    a0, p0 = attend(model.blocks[0], r0, l0_override)
    r1 = r0 + a0
    a1, p1 = attend(model.blocks[1], r1, l1_override)
    r2 = r1 + a1
    return dict(logits=model.unembed(model.ln_final(r2))[0], embed=r0[0], L0=a0[0], L1=a1[0],
                final=r2[0], p0=p0[0], p1=p1[0], r1=r1[0])

print("1. Clean count test: delete the Y block from layer 0's average at position i (renormalise).")
print("   Consumed symbols should go back to looking valid.\n")
rng = random.Random(11)
for K in [4, 6]:
    before, after = [], []
    for _ in range(80):
        toks = make_prompt(rng, K); i = 2 * K   # second to last: K-1 consumed, 1 remaining
        t = run(toks); Xs = toks[1:K + 1]; consumed = set(toks[K + 2:i + 1])
        row = t["p0"][i].clone(); row[K + 2:] = 0; row = row / row.sum()
        t2 = run(toks, l0_override={i: row})
        for s in Xs:
            (before if s in consumed else after).append(0)
        cons = [s for s in Xs if s in consumed]
        rem = [s for s in Xs if s not in consumed]
        before.append((float(t["logits"][i, cons].mean()), float(t["logits"][i, rem].mean())))
        after.append((float(t2["logits"][i, cons].mean()), float(t2["logits"][i, rem].mean())))
    b = np.array([x for x in before if isinstance(x, tuple)]); a = np.array([x for x in after if isinstance(x, tuple)])
    print(f"  K={K} normal:      consumed {b[:,0].mean():6.2f}   remaining {b[:,1].mean():6.2f}")
    print(f"  K={K} Y deleted:   consumed {a[:,0].mean():6.2f}   remaining {a[:,1].mean():6.2f}   -> gap {b[:,1].mean()-b[:,0].mean():.2f} collapses to {a[:,1].mean()-a[:,0].mean():.2f}")

print("\n2. Why does layer 1 leave SEP at the end? Content or position?")
print("   Off-distribution probe: repeat a symbol in Y so the LAST position still has something remaining.")
for K in [4, 6]:
    valid_last, invalid_last = [], []
    for _ in range(80):
        X = rng.sample(range(NUM_SYMBOLS), K)
        Y = X[:]; rng.shuffle(Y)
        toks = encode(X, Y); i = 2 * K + 1
        t = run(toks); valid_last.append((float(t["p1"][i, K + 1]), float(t["logits"][i, EOS])))
        Yb = Y[:]; Yb[-1] = Yb[0]                       # repeat, so one symbol never appears
        tb = run(encode(X, Yb))
        invalid_last.append((float(tb["p1"][i, K + 1]), float(tb["logits"][i, EOS])))
    v, w = np.array(valid_last), np.array(invalid_last)
    print(f"  K={K}: same position, all consumed -> SEP mass {v[:,0].mean():.3f}, EOS logit {v[:,1].mean():7.2f}")
    print(f"  K={K}: same position, one left     -> SEP mass {w[:,0].mean():.3f}, EOS logit {w[:,1].mean():7.2f}")

print("\n3. Layer 1 attention on SEP as a function of how many symbols remain (all positions pooled)")
for K in [5, 8]:
    byrem = {}
    for _ in range(150):
        toks = make_prompt(rng, K); t = run(toks); Xs = set(toks[1:K + 1])
        for i in range(K + 1, 2 * K + 2):
            r = len(Xs - set(toks[K + 2:i + 1]))
            byrem.setdefault(r, []).append(float(t["p1"][i, K + 1]))
    print(f"  K={K}: " + " ".join(f"rem {r}: {np.mean(v):.3f}" for r, v in sorted(byrem.items())))

print("\n4. The layer-1 OV circuit: does reading SEP copy the set X into the logits?")
print("   W_U_eff = (U*g) @ W_O^1 @ W_V^1 @ RMSNorm-ish, applied to each symbol embedding:")
b1 = model.blocks[1]
OV1 = b1.attn.W_O.weight @ b1.attn.W_V.weight
M = (Uf @ OV1) @ (E * b1.ln_attn.weight).T          # (19 output tokens, 19 attended tokens)
M = M[:NUM_SYMBOLS, :NUM_SYMBOLS]
print(f"   mean diagonal {M.diag().mean():8.1f}   mean off-diagonal {((M.sum()-M.diag().sum())/(16*15)):8.1f}   ratio {M.diag().mean()/abs((M.sum()-M.diag().sum())/(16*15)):.1f}x")
print("   -> reading a symbol's vector pushes that symbol's own logit up. It is a copy circuit.")
b0 = model.blocks[0]
OV0 = b0.attn.W_O.weight @ b0.attn.W_V.weight
M0 = (Uf @ OV0) @ (E * b0.ln_attn.weight).T
M0 = M0[:NUM_SYMBOLS, :NUM_SYMBOLS]
print(f"   layer 0 OV: mean diagonal {M0.diag().mean():8.1f}   mean off-diagonal {((M0.sum()-M0.diag().sum())/(16*15)):8.1f}")
print("   -> layer 0's copy circuit has the opposite sign. It subtracts each symbol it sees.")

print("\n5. Putting it together: predict the logit from counts alone.")
print("   model:  logit(s) = alpha_K * [s in X] - beta_K * (times s has appeared) + const")
for K in [3, 5, 8]:
    rows, ys = [], []
    for _ in range(60):
        toks = make_prompt(rng, K); t = run(toks); Xs = set(toks[1:K + 1])
        for i in range(K + 1, 2 * K + 2):
            consumed = set(toks[K + 2:i + 1])
            for s in range(NUM_SYMBOLS):
                rows.append([1.0 if s in Xs else 0.0, (1 if s in Xs else 0) + (1 if s in consumed else 0), 1.0])
                ys.append(float(t["logits"][i, s]))
    A = np.array(rows); y = np.array(ys)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  K={K}: alpha {coef[0]:6.2f}  beta {-coef[1]:6.2f}  const {coef[2]:6.2f}   R^2 = {r2:.3f}")
