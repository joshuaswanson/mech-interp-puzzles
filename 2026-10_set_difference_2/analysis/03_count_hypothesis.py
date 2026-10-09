import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()

print("Hypothesis: the logit of symbol s depends on how many times s has appeared so far.")
print("count 0 = not in X; count 1 = in X, not yet consumed (VALID); count 2 = in X and consumed (INVALID).\n")
rng = random.Random(2)
B = batch(rng, 150)
for K, x in B.items():
    logits, _ = model(x)
    buckets = {0: [], 1: [], 2: []}
    for b in range(len(x)):
        toks = x[b].tolist()
        X = set(toks[1:K + 1])
        for i in range(K + 1, 2 * K + 2):
            consumed = set(toks[K + 2:i + 1])
            for s in range(NUM_SYMBOLS):
                c = (1 if s in X else 0) + (1 if s in consumed else 0)
                buckets[c].append(float(logits[b, i, s]))
    print(f"K={K}: " + "  ".join(f"count {c}: mean {np.mean(v):7.2f} sd {np.std(v):5.2f} range [{np.min(v):7.2f},{np.max(v):7.2f}]" for c, v in buckets.items()))

print("\nSeparation: min logit of a count-1 symbol minus max logit of any count-0 or count-2 symbol, per position")
worst = 1e9
for K, x in B.items():
    logits, _ = model(x)
    m = []
    for b in range(len(x)):
        toks = x[b].tolist(); X = set(toks[1:K + 1])
        for i in range(K + 1, 2 * K + 1):
            consumed = set(toks[K + 2:i + 1])
            valid = [s for s in range(NUM_SYMBOLS) if s in X and s not in consumed]
            other = [s for s in range(NUM_SYMBOLS) if s not in valid] + [EOS]
            if valid:
                m.append(float(logits[b, i, valid].min() - logits[b, i, other].max()))
    print(f"  K={K}: min margin {min(m):6.2f}  mean {np.mean(m):6.2f}")
    worst = min(worst, min(m))
print(f"  worst margin overall: {worst:.2f}")

print("\nDoes a symbol's logit depend on WHERE it sits in X or Y, or only on its count?")
print("logit of count-1 symbols grouped by their position index within X:")
for K, x in list(B.items())[3:4]:
    logits, _ = model(x)
    byslot = {}
    for b in range(len(x)):
        toks = x[b].tolist(); X = toks[1:K + 1]
        for i in range(K + 1, 2 * K + 2):
            consumed = set(toks[K + 2:i + 1])
            for slot, s in enumerate(X):
                if s not in consumed:
                    byslot.setdefault(slot, []).append(float(logits[b, i, s]))
    print(f"  K={K}: " + " ".join(f"slot {k}: {np.mean(v):6.2f}" for k, v in sorted(byslot.items())))

print("\nEOS logit by number of symbols still remaining:")
for K, x in B.items():
    logits, _ = model(x)
    byrem = {}
    for b in range(len(x)):
        toks = x[b].tolist(); X = set(toks[1:K + 1])
        for i in range(K + 1, 2 * K + 2):
            consumed = set(toks[K + 2:i + 1])
            r = len(X - consumed)
            byrem.setdefault(r, []).append(float(logits[b, i, EOS]))
    print(f"  K={K}: " + " ".join(f"rem {r}: {np.mean(v):7.2f}" for r, v in sorted(byrem.items())))
