import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
rng = np.random.default_rng(0)
x, counts = random_batch(rng, 300)
pred, logits, attns = predict(model, x)
print("accuracy:", (pred == counts).float().mean().item())

E = model.tok_embed.weight
U = model.unembed.weight
print("\nEmbedding norms:", E.norm(dim=1))
print("Unembed norms  :", U.norm(dim=1))

labels = [token_label(t) for t in range(VOCAB_SIZE)]
En = E / E.norm(dim=1, keepdim=True)
print("\nEmbedding cosine similarity (symbols a..j, BOS, ANS):")
cos = En[:12] @ En[:12].T
print("       " + " ".join(f"{l:>6}" for l in labels[:12]))
for i in range(12):
    print(f"{labels[i]:>6} " + " ".join(f"{v:6.2f}" for v in cos[i].tolist()))

print("\nDirect path logits (E @ U^T) for ANS token over count outputs:")
print((E[ANS] @ U[COUNT_BASE:].T))
