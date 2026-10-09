import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()
E = model.tok_embed.weight
U = model.unembed.weight
g_final = model.ln_final.weight
Uf = U * g_final                      # logit_s = (U_s * g) . r / rms(r)

def rms(v):
    return (v.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()

def trace(tokens):
    """Exact decomposition of the final residual into embed / layer0 / layer1 parts."""
    x = torch.tensor([tokens])
    n = len(tokens)
    r0 = E[x]                                             # (1, n, 64)
    rope = mod.rope_cos_sin(n, 64, 10000.0, x.device)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    b0 = model.blocks[0]
    a0_out, p0 = b0.attn(b0.ln_attn(r0), mask, rope)
    r1 = r0 + a0_out
    b1 = model.blocks[1]
    a1_out, p1 = b1.attn(b1.ln_attn(r1), mask, rope)
    r2 = r1 + a1_out
    logits = model.unembed(model.ln_final(r2))
    return dict(embed=r0[0], L0=a0_out[0], L1=a1_out[0], final=r2[0], logits=logits[0], p0=p0[0, 0], p1=p1[0, 0])

rng = random.Random(3)
print("Per-component contribution to the logit of a symbol, bucketed by its count so far.")
print("logit_s = [ (U_s*g) . embed + (U_s*g) . L0 + (U_s*g) . L1 ] / rms(final)\n")
for K in [3, 5, 8]:
    acc = {c: {"embed": [], "L0": [], "L1": [], "total": []} for c in (0, 1, 2)}
    for _ in range(120):
        toks = make_prompt(rng, K)
        t = trace(toks)
        Xs = set(toks[1:K + 1])
        for i in range(K + 1, 2 * K + 2):
            consumed = set(toks[K + 2:i + 1])
            scale = float(rms(t["final"][i]))
            for s in range(NUM_SYMBOLS):
                c = (1 if s in Xs else 0) + (1 if s in consumed else 0)
                for part in ("embed", "L0", "L1"):
                    acc[c][part].append(float(Uf[s] @ t[part][i]) / scale)
                acc[c]["total"].append(float(t["logits"][i, s]))
    print(f"K={K}")
    for c in (0, 1, 2):
        a = acc[c]
        print(f"  count {c}: embed {np.mean(a['embed']):7.2f}   L0 {np.mean(a['L0']):7.2f}   L1 {np.mean(a['L1']):7.2f}   total {np.mean(a['total']):7.2f}")
    print(f"    -> L0 drop from count 1 to count 2: {np.mean(acc[2]['L0']) - np.mean(acc[1]['L0']):6.2f};  L1 gain for being in X: {np.mean(acc[1]['L1']) - np.mean(acc[0]['L1']):6.2f}")

print("\n\nLayer 0 attention: is it a uniform average over tokens seen so far?")
for K in [3, 5, 8]:
    toks = make_prompt(random.Random(7), K); t = trace(toks)
    for i in [K + 1, K + 3, 2 * K + 1]:
        row = t["p0"][i, :i + 1]
        sym = [j for j in range(i + 1) if toks[j] < NUM_SYMBOLS]
        print(f"  K={K} pos {i:2d}: on symbol tokens {float(row[sym].sum()):.3f} (each {float(row[sym].mean()):.3f}, sd {float(row[sym].std()):.4f}), BOS {float(row[0]):.3f}, SEP {float(row[K+1]):.3f}")

print("\n\nLayer 1 attention at scored positions: mass on SEP")
B = batch(random.Random(4), 100)
for K, x in B.items():
    logits, attns = model(x)
    A = attns[1][:, 0]
    m = [float(A[:, i, K + 1].mean()) for i in range(K + 1, 2 * K + 2)]
    print(f"  K={K}: " + " ".join(f"i={j}:{v:.3f}" for j, v in enumerate(m)))

print("\n\nWhy EOS fires when nothing remains: rms of the final residual vs symbols remaining")
for K in [3, 5, 8]:
    byrem = {}
    for _ in range(120):
        toks = make_prompt(rng, K); t = trace(toks)
        Xs = set(toks[1:K + 1])
        for i in range(K + 1, 2 * K + 2):
            r = len(Xs - set(toks[K + 2:i + 1]))
            byrem.setdefault(r, []).append((float(rms(t["final"][i])), float(t["logits"][i, EOS])))
    print(f"  K={K}: " + " ".join(f"rem {r}: rms {np.mean([a for a,_ in v]):5.2f} eos {np.mean([b for _,b in v]):7.2f}" for r, v in sorted(byrem.items())))
