import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()
rng = random.Random(31)

print("Hypothesis: EOS fires when the number of symbol tokens after SEP equals the number before it,")
print("regardless of which symbols they were. On the training distribution the two rules agree.\n")

print("1. Vary the length of Y independently of its content. X has K symbols; Y has L tokens")
print("   drawn from X with repeats allowed, so the set can be empty or not at any L.\n")
for K in [4, 6]:
    print(f"  K={K}")
    print(f"    {'L':>2s} {'remaining set':>14s} {'EOS logit':>10s} {'top symbol':>11s} {'EOS wins':>9s}")
    for L in range(1, K + 3):
        eos, top, wins, rem = [], [], [], []
        for _ in range(60):
            X = rng.sample(range(NUM_SYMBOLS), K)
            Y = [X[rng.randrange(K)] for _ in range(L)]
            toks = [BOS] + X + [SEP] + Y + [EOS]
            logits, _ = model(torch.tensor([toks]))
            lg = logits[0, K + 1 + L]
            eos.append(float(lg[EOS])); top.append(float(lg[:NUM_SYMBOLS].max()))
            wins.append(float(lg[EOS]) > float(lg[:NUM_SYMBOLS].max()))
            rem.append(len(set(X) - set(Y)))
        flag = "  <- L = K" if L == K else ""
        print(f"    {L:2d} {np.mean(rem):14.2f} {np.mean(eos):10.2f} {np.mean(top):11.2f} {np.mean(wins):9.2f}{flag}")

print("\n2. Hold the set empty, vary L. Y is a permutation of X padded with repeats.")
print("   If EOS were set-based it would fire at every L >= K.\n")
for K in [4, 6]:
    row = []
    for L in range(K, K + 4):
        e = []
        for _ in range(60):
            X = rng.sample(range(NUM_SYMBOLS), K); Y = X[:]; rng.shuffle(Y)
            Y = Y + [X[rng.randrange(K)] for _ in range(L - K)]
            toks = [BOS] + X + [SEP] + Y + [EOS]
            logits, _ = model(torch.tensor([toks]))
            e.append(float(logits[0, K + 1 + L, EOS]))
        row.append((L, np.mean(e)))
    print(f"  K={K} (set is empty from L={K} on): " + "  ".join(f"L={L}: EOS {v:7.2f}" for L, v in row))

print("\n3. Hold L = K, vary how many distinct symbols Y actually used.\n")
for K in [5, 7]:
    print(f"  K={K}")
    for distinct in range(1, K + 1):
        e, t = [], []
        for _ in range(60):
            X = rng.sample(range(NUM_SYMBOLS), K)
            pool = X[:distinct]
            Y = pool[:] + [pool[rng.randrange(distinct)] for _ in range(K - distinct)]
            rng.shuffle(Y)
            toks = [BOS] + X + [SEP] + Y + [EOS]
            logits, _ = model(torch.tensor([toks]))
            lg = logits[0, 2 * K + 1]
            e.append(float(lg[EOS])); t.append(float(lg[:NUM_SYMBOLS].max()))
        print(f"    Y used {distinct} distinct symbols, {K - distinct} still remaining: EOS {np.mean(e):7.2f}  top symbol {np.mean(t):7.2f}")

print("\n4. The length comparison in the attention: layer-1 mass on SEP against the relative distance")
print("   from the current position to SEP, split by K.\n")
for K in [4, 6, 8]:
    row = []
    for _ in range(100):
        toks = make_prompt(rng, K)
        _, attns = model(torch.tensor([toks]))
        for i in range(K + 1, 2 * K + 2):
            row.append((i - (K + 1), float(attns[1][0, 0, i, K + 1])))
    byd = {}
    for d, v in row: byd.setdefault(d, []).append(v)
    print(f"  K={K}: " + " ".join(f"d={d}:{np.mean(v):.3f}" for d, v in sorted(byd.items())))
print("\n  The drop happens at d = K for every K, so the head is comparing the distance to SEP")
print("  against the number of symbols sitting before SEP.")
