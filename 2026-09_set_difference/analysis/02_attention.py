import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load_model()
E, P, U, L = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight, m.layers[0]
ORDER = "dahgbkefionplcjm"

examples = [
    (["n", "b", "j", "e"], ["e", "n", "b"]),
    (["a", "c", "k", "p"], ["k", "p", "c"]),
    (["d", "h", "l", "m"], ["h", "d", "l"]),
    (["m", "j", "c", "a"], ["m", "j", "c"]),
    (["d", "a", "h", "g"], ["a", "g", "h"]),
]
for X, Y in examples:
    Xi = [ord(c) - 97 for c in X]; Yi = [ord(c) - 97 for c in Y]
    z = [s for s in Xi if s not in Yi][0]
    toks = encode(Xi, Yi)
    logits, attn = m(torch.tensor([toks]))
    labs = [label(t) for t in toks]
    pred = label(int(logits[0, -1, :16].argmax()))
    print(f"\n{' '.join(labs)}   missing = {label(z)}   pred = {pred}")
    for h in range(2):
        a = attn[0][0, h, -1]
        print(f"  head {h} attention from final SEP: " + " ".join(f"{labs[j]}:{a[j]:.2f}" for j in range(10)))

# Per-token value and per-position value for each head
print("\nPer-token value W_V.E[t] and per-position value W_V.P[j] for each head")
for h in range(2):
    wv = L.heads[h].W_V.weight[0]
    print(f"  head {h}: tokens  " + " ".join(f"{label(t)}={float(wv @ E[t]):6.1f}" for t in range(18)))
    print(f"          positions " + " ".join(f"{j}={float(wv @ P[j]):6.1f}" for j in range(10)))
print("\nQuery at final SEP and key of each token/position part for each head")
r9 = E[SEP] + P[9]
for h in range(2):
    hd = L.heads[h]
    q = float(hd.W_Q.weight[0] @ r9)
    wk = hd.W_K.weight[0]
    print(f"  head {h}: q = {q:.3f};  score = q * k, k = W_K.r")
    print("     token part of k: " + " ".join(f"{label(t)}={float(wk @ E[t]):6.2f}" for t in range(18)))
    print("     position part of k: " + " ".join(f"{j}={float(wk @ P[j]):6.2f}" for j in range(10)))
