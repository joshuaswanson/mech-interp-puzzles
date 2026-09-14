import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
rng = np.random.default_rng(3)
x, counts = random_batch(rng, 300)
_, logits, attns = predict(model, x)
A1 = attns[1][:, :, -1, :]  # (B, 4, 12) ANS-row attention in layer 1
A0 = attns[0][:, :, -1, :]

seqs = x[:, 1:-1]  # (B, 10)
B = len(counts)
# first-occurrence mask
first = torch.zeros(B, SEQ_LEN, dtype=torch.bool)
for b in range(B):
    seen = set()
    for i, t in enumerate(seqs[b].tolist()):
        first[b, i] = t not in seen
        seen.add(t)

for L, A in [(0, A0), (1, A1)]:
    print(f"\n===== Layer {L} ANS-row attention, averaged per count =====")
    print(f"{'head':5s} {'count':5s} {'BOS':>6s} {'ANS':>6s} {'first':>6s} {'repeat':>6s} | first-per-unique  repeat-per-repeat")
    for h in range(4):
        for c in range(1, 11):
            m = counts == c
            a = A[m, h]
            bos = a[:, 0].mean().item()
            ans = a[:, -1].mean().item()
            f = (a[:, 1:-1] * first[m]).sum(-1).mean().item()
            r = (a[:, 1:-1] * ~first[m]).sum(-1).mean().item()
            fpu = f / c
            rpr = r / (10 - c) if c < 10 else float("nan")
            print(f"L{L}H{h}  {c:5d} {bos:6.3f} {ans:6.3f} {f:6.3f} {r:6.3f} | {fpu:8.4f}          {rpr:8.4f}")
        print()

print("\n===== Layer 1 ANS-row attention on symbol positions, by symbol identity (avg weight per occurrence) =====")
for h in range(4):
    per_sym = []
    for s in range(NUM_SYMBOLS):
        m = seqs == s
        per_sym.append((A1[:, h, 1:-1][m]).mean().item())
    print(f"L1H{h}: " + " ".join(f"{chr(97+s)}={v:.3f}" for s, v in enumerate(per_sym)))
print("\n===== Layer 0 ANS-row attention on symbol positions, by symbol identity (avg weight per occurrence) =====")
for h in range(4):
    per_sym = []
    for s in range(NUM_SYMBOLS):
        m = seqs == s
        per_sym.append((A0[:, h, 1:-1][m]).mean().item())
    print(f"L0H{h}: " + " ".join(f"{chr(97+s)}={v:.3f}" for s, v in enumerate(per_sym)))
