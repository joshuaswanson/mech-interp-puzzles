import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight

def qk_matrix(layer, head):
    h = model.layers[layer].heads[head]
    return (h.W_Q(E) @ h.W_K(E).T) / (D_HEAD ** 0.5)

print("SVD singular values of layer-0 QK matrices (12x12, symbols+BOS+ANS):")
for head in range(4):
    S = qk_matrix(0, head)[:12, :12]
    sv = torch.linalg.svdvals(S)
    print(f"  L0H{head}: {sv[:5].numpy()}")

examples = [
    "abacabdace",
    "aaaaaaaaaa",
    "ababababab",
    "abcdefghij",
    "abcdeabcde",
]
for s in examples:
    x = torch.tensor([encode(list(s))])
    pred, logits, attns = predict(model, x)
    labels = [token_label(t) for t in x[0].tolist()]
    print(f"\n\n######## {s}  true={len(set(s))} pred={pred.item()}")
    for L in range(2):
        for h in range(4):
            A = attns[L][0, h]
            print(f"\n-- Layer {L} Head {h} --   (rows=query pos, cols=key pos)")
            print("      " + " ".join(f"{l:>5}" for l in labels))
            for i in range(12):
                row = " ".join(f"{v:5.2f}" if j <= i else "     " for j, v in enumerate(A[i].tolist()))
                print(f"{labels[i]:>5} " + row)
