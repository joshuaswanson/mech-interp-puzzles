import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight
labels = [token_label(t) for t in range(12)]

def qk_matrix(layer, head):
    h = model.layers[layer].heads[head]
    q = h.W_Q(E)
    k = h.W_K(E)
    return (q @ k.T) / (D_HEAD ** 0.5)

for head in range(4):
    S = qk_matrix(0, head)[:12, :12]
    print(f"\n=== Layer 0 Head {head}: QK scores (rows=query token, cols=key token) ===")
    print("       " + " ".join(f"{l:>6}" for l in labels))
    for i in range(12):
        print(f"{labels[i]:>6} " + " ".join(f"{v:6.2f}" for v in S[i].tolist()))
    diag = S.diag()[:10]
    offdiag = S[:10, :10] - torch.diag(S.diag()[:10])
    print(f"  mean diag (sym->same sym): {diag.mean():.2f}   mean offdiag (sym->other sym): {(offdiag.sum()/90):.2f}")
    print(f"  sym->BOS: {S[:10, BOS].mean():.2f} (std {S[:10, BOS].std():.2f})   ANS->sym: {S[ANS, :10].mean():.2f} (std {S[ANS, :10].std():.2f})   ANS->BOS: {S[ANS, BOS]:.2f}   ANS->ANS: {S[ANS, ANS]:.2f}")
