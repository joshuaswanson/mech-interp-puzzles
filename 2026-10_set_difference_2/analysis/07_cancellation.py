import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()
E, U = model.tok_embed.weight, model.unembed.weight
Uf = U * model.ln_final.weight
b0, b1 = model.blocks[0], model.blocks[1]

print("1. OV circuits, with scale. Entry [s, t] = how much reading token t at that layer moves symbol s's logit.")
for name, blk in [("layer 0", b0), ("layer 1", b1)]:
    OV = blk.attn.W_O.weight @ blk.attn.W_V.weight
    M = ((Uf @ OV) @ (E * blk.ln_attn.weight).T)[:NUM_SYMBOLS, :NUM_SYMBOLS]
    d = M.diag(); off = (M.sum() - d.sum()) / (NUM_SYMBOLS * (NUM_SYMBOLS - 1))
    print(f"  {name}: diagonal mean {d.mean():+9.3f} (sd {d.std():.3f})   off-diagonal mean {off:+9.3f}   diagonal/|off| = {abs(d.mean()/off):5.1f}")
print("  Layer 1 reads a symbol and raises that symbol's own logit. Layer 0 reads a symbol and lowers it.")
print("  The two copy circuits have opposite sign, which is what makes occurrences cancel.\n")

print("2. The cancellation condition. Fit logit(s) = alpha*[s in X] - beta*(count of s) + c, per K.")
print("   A symbol seen twice scores alpha - 2*beta, so exact cancellation needs alpha = 2*beta.\n")
rng = random.Random(21)
print(f"  {'K':>2s} {'alpha':>7s} {'beta':>7s} {'2*beta':>7s} {'alpha-2beta':>12s} {'count-2 logit':>14s} {'count-1 logit':>14s} {'R^2':>6s}")
for K in range(2, 9):
    rows, ys, c1, c2 = [], [], [], []
    for _ in range(80):
        toks = make_prompt(rng, K)
        logits, _ = model(torch.tensor([toks]))
        Xs = set(toks[1:K + 1])
        for i in range(K + 1, 2 * K + 2):
            consumed = set(toks[K + 2:i + 1])
            for s in range(NUM_SYMBOLS):
                c = (1 if s in Xs else 0) + (1 if s in consumed else 0)
                rows.append([1.0 if s in Xs else 0.0, float(c), 1.0]); ys.append(float(logits[0, i, s]))
                if c == 1: c1.append(float(logits[0, i, s]))
                if c == 2: c2.append(float(logits[0, i, s]))
    A, y = np.array(rows), np.array(ys)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((A @ coef - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    a, b = coef[0], -coef[1]
    print(f"  {K:2d} {a:7.2f} {b:7.2f} {2*b:7.2f} {a-2*b:12.2f} {np.mean(c2):14.2f} {np.mean(c1):14.2f} {r2:6.3f}")

print("\n3. The EOS trigger, tested off distribution.")
print("   Build Y with one symbol repeated, so at the final position one symbol of X was never consumed.")
print("   If EOS were purely positional it would still win. If it tracks the set, the leftover symbol should win.\n")
print(f"  {'K':>2s} {'case':>24s} {'EOS logit':>10s} {'best symbol logit':>18s} {'winner':>8s}")
for K in [4, 6, 8]:
    for case in ("all consumed", "one never consumed"):
        eos, best, win = [], [], []
        for _ in range(80):
            X = rng.sample(range(NUM_SYMBOLS), K); Y = X[:]; rng.shuffle(Y)
            if case == "one never consumed":
                Y[-1] = Y[0]
            toks = encode(X, Y) if case == "all consumed" else [BOS] + X + [SEP] + Y + [EOS]
            logits, _ = model(torch.tensor([toks]))
            lg = logits[0, 2 * K + 1]
            left = [s for s in X if s not in set(Y)]
            eos.append(float(lg[EOS]))
            best.append(float(lg[left].max()) if left else float(lg[:NUM_SYMBOLS].max()))
            win.append("EOS" if float(lg[EOS]) > (float(lg[left].max()) if left else float(lg[:NUM_SYMBOLS].max())) else "symbol")
        print(f"  {K:2d} {case:>24s} {np.mean(eos):10.2f} {np.mean(best):18.2f} {max(set(win), key=win.count):>8s}")

print("\n4. Same probe, but on the layer-1 attention: does it leave SEP because the set is empty,")
print("   or because the position is last?\n")
for K in [4, 8]:
    for case in ("all consumed", "one never consumed"):
        m = []
        for _ in range(80):
            X = rng.sample(range(NUM_SYMBOLS), K); Y = X[:]; rng.shuffle(Y)
            if case == "one never consumed": Y[-1] = Y[0]
            toks = [BOS] + X + [SEP] + Y + [EOS]
            _, attns = model(torch.tensor([toks]))
            m.append(float(attns[1][0, 0, 2 * K + 1, K + 1]))
        print(f"  K={K} {case:>22s}: layer-1 mass on SEP at the final position = {np.mean(m):.3f}")
