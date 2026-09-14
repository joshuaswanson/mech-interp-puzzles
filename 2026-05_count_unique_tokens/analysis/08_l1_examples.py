import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
rng = np.random.default_rng(5)
x, counts = random_batch(rng, 4)
_, logits, attns = predict(model, x)
A1 = attns[1][:, :, -1, :]

for b in range(len(counts)):
    s = "".join(chr(97 + t) for t in x[b, 1:-1].tolist())
    line = f"{s} U={counts[b].item():2d} | "
    for h in range(4):
        a = A1[b, h]
        top = torch.argsort(a, descending=True)[:3]
        parts = []
        for j in top.tolist():
            if a[j] < 0.02:
                continue
            lab = "BOS" if j == 0 else ("ANS" if j == 11 else f"{chr(97 + x[b, j].item())}@{j}")
            parts.append(f"{lab}:{a[j]:.2f}")
        line += f"H{h}[{' '.join(parts)}]  "
    print(line)
