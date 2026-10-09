import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()

def show(tokens, title):
    x = torch.tensor([tokens])
    logits, attns = model(x)
    labs = [label(t) for t in tokens]
    K = (len(tokens) - 3) // 2
    print(f"\n########## {title}:  {' '.join(labs)}   (K={K})")
    for L in range(2):
        A = attns[L][0, 0]
        print(f"\n-- layer {L} attention (rows = query, cols = key) --")
        print("      " + " ".join(f"{l:>4}" for l in labs))
        for i in range(len(tokens)):
            row = " ".join(f"{A[i,j]:4.2f}" if j <= i else "    " for j in range(len(tokens)))
            mark = " <- scored" if K + 1 <= i <= 2 * K + 1 else ""
            print(f"{labs[i]:>4}{i:>2} " + row + mark)

rng = random.Random(0)
show(encode([1, 4, 9, 13, 2], [1, 9, 4, 13, 2]), "K=5 example")
show(encode([0, 5], [5, 0]), "K=2 example")
show(encode([0, 3, 5, 7, 10, 12, 14, 15], [15, 3, 0, 14, 5, 12, 7, 10]), "K=8 example")

# Quantify: where does each layer's attention go, by query role?
print("\n\n########## aggregate attention mass by query role and key role ##########")
B = batch(random.Random(1), 200)
for L in range(2):
    print(f"\n--- layer {L} ---")
    print(f"{'query role':16s} {'BOS':>6s} {'X: matched':>11s} {'X: unused':>10s} {'SEP':>6s} {'Y: earlier':>11s} {'Y: self':>8s}")
    for K, x in B.items():
        logits, attns = model(x)
        A = attns[L][:, 0]
        acc = {}
        for b in range(len(x)):
            toks = x[b].tolist()
            Xpos = list(range(1, K + 1)); Ypos = list(range(K + 2, 2 * K + 2))
            for i in range(K + 1, 2 * K + 2):
                consumed = set(toks[K + 2:i + 1])
                role = "SEP" if i == K + 1 else f"y{i - K - 1}"
                bos = float(A[b, i, 0])
                xm = sum(float(A[b, i, j]) for j in Xpos if toks[j] in consumed)
                xu = sum(float(A[b, i, j]) for j in Xpos if toks[j] not in consumed)
                sep = float(A[b, i, K + 1])
                ye = sum(float(A[b, i, j]) for j in Ypos if j < i)
                ys = float(A[b, i, i]) if i in Ypos else 0.0
                key = (K, role)
                a = acc.setdefault(key, np.zeros(6)); a += np.array([bos, xm, xu, sep, ye, ys]); 
                acc[key] = a
        for (KK, role), a in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            a = a / len(x)
            print(f"K={KK} {role:10s} " + " ".join(f"{v:6.3f}" for v in [a[0], a[1], a[2], a[3], a[4], a[5]]))
        break  # one K at a time is enough; loop prints K=2 first
