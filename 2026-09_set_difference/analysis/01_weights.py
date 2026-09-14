import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load_model()
E = m.tok_embed.weight
P = m.pos_embed.weight
U = m.unembed.weight
L = m.layers[0]
print("parameter count:", sum(p.numel() for p in m.parameters()))
print("\nToken embeddings E (18 x 2):")
for t in range(18):
    r = E[t]
    print(f"  {label(t):>3s}: ({r[0]:7.3f}, {r[1]:7.3f})   norm {r.norm():.3f}  angle {np.degrees(np.arctan2(r[1], r[0])):7.1f}")
print("\nPositional embeddings P (10 x 2):")
for j in range(10):
    r = P[j]
    print(f"  pos {j}: ({r[0]:7.3f}, {r[1]:7.3f})   norm {r.norm():.3f}")
for h in range(2):
    hd = L.heads[h]
    print(f"\nHead {h}: W_Q = {hd.W_Q.weight.numpy().round(3)}, W_K = {hd.W_K.weight.numpy().round(3)}, W_V = {hd.W_V.weight.numpy().round(3)}")
print("\nW_O (2 x 2, columns = heads):")
print(L.W_O.weight.numpy().round(3))
print("\nUnembed U (18 x 2):")
for t in range(18):
    r = U[t]
    print(f"  {label(t):>3s}: ({r[0]:7.3f}, {r[1]:7.3f})   norm {r.norm():.3f}  angle {np.degrees(np.arctan2(r[1], r[0])):7.1f}")

# accuracy on a large random sample and exhaustive per-set correctness on all sets with one ordering
prompts = all_prompts(max_n=200000, seed=1)
x = torch.tensor([p for p, _ in prompts]); z = torch.tensor([zz for _, zz in prompts])
logits, attn = m(x)
pred = logits[:, -1, :NUM_SYMBOLS].argmax(-1)
print(f"\naccuracy on {len(prompts)} random orderings: {(pred == z).float().mean():.5f}")
