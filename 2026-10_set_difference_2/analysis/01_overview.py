import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()
print("config:", json.dumps(config["model"]))
print("parameters:", sum(p.numel() for p in model.parameters()))
print("\nweights:")
for name, p in model.named_parameters():
    print(f"  {name:32s} {tuple(p.shape)}")

rng = random.Random(0)
B = batch(rng, 400)
print("\nseparation metric: at every scored position, does every valid token outrank every invalid one?")
tot = ok = 0
for K, x in B.items():
    logits, attns = model(x)
    good = torch.ones(len(x), dtype=torch.bool)
    for b in range(len(x)):
        for pos, valid in scored_positions(x[b].tolist()).items():
            lg = logits[b, pos]
            mask = torch.zeros(19, dtype=torch.bool); mask[valid] = True
            if lg[mask].min() <= lg[~mask].max():
                good[b] = False
    tot += len(x); ok += int(good.sum())
    print(f"  K={K}: {good.float().mean():.4f}")
print(f"  overall: {ok}/{tot}")

print("\nmean probability mass on the valid set, by K and by step i (how many symbols already consumed)")
for K, x in B.items():
    logits, _ = model(x)
    probs = logits.softmax(-1)
    row = []
    for i in range(K + 1):
        pos = K + 1 + i
        m = torch.zeros(len(x), 19, dtype=torch.bool)
        for b in range(len(x)):
            m[b, scored_positions(x[b].tolist())[pos]] = True
        row.append(float((probs[:, pos] * m).sum(-1).mean()))
    print(f"  K={K}: " + " ".join(f"i={i}:{v:.3f}" for i, v in enumerate(row)))

print("\nuniformity: ratio of max to min probability within the valid set (1.0 = perfectly uniform)")
for K, x in B.items():
    logits, _ = model(x)
    probs = logits.softmax(-1)
    rows = []
    for i in range(K):
        pos = K + 1 + i
        r = []
        for b in range(len(x)):
            v = scored_positions(x[b].tolist())[pos]
            if len(v) > 1:
                p = probs[b, pos, v]
                r.append(float(p.max() / p.min()))
        if r: rows.append((i, np.mean(r)))
    print(f"  K={K}: " + " ".join(f"i={i}:{v:.2f}" for i, v in rows))
