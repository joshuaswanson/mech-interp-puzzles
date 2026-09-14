import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight
U = model.unembed.weight[COUNT_BASE:]
rng = np.random.default_rng(12)
x, counts = random_batch(rng, 400)
b, s = x.shape
h = E[x]
mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
for layer in model.layers:
    outs = [head(h, mask)[0] for head in layer.heads]
    h = h + layer.W_O(torch.cat(outs, -1))
r = h[:, -1]
# fit r ~ U g + r0
X1 = torch.stack([counts.float(), torch.ones(len(counts))], 1)
W = torch.linalg.lstsq(X1, r).solution
g, r0 = W[0], W[1]
res = r - X1 @ W
print(f"Residual at ANS ~ U*g + r0:  R^2 = {1 - (res**2).sum() / ((r - r.mean(0))**2).sum():.3f}")
print(f"|g| = {g.norm():.3f}, |r0| = {r0.norm():.3f}, mean |residual| = {res.norm(dim=1).mean():.3f}")
c = torch.arange(1, 11).float()
print("\nU_c . g   (should be ~linear in c):", (U @ g).numpy().round(1))
print("U_c . r0  (should be ~concave quadratic):", (U @ r0).numpy().round(1))
A = torch.stack([c**2, c, torch.ones(10)], 1)
cg = torch.linalg.lstsq(A, (U @ g).unsqueeze(1)).solution.squeeze()
cr = torch.linalg.lstsq(A, (U @ r0).unsqueeze(1)).solution.squeeze()
print(f"\nU_c.g  ~ {cg[0]:.2f} c^2 + {cg[1]:.2f} c + {cg[2]:.2f}")
print(f"U_c.r0 ~ {cr[0]:.2f} c^2 + {cr[1]:.2f} c + {cr[2]:.2f}")
print(f"=> logit_c ~ {cg[1]:.1f} U c {cr[0]:+.1f} c^2 {cr[1]:+.1f} c + const;  argmax at c = {cg[1]:.1f} U / {2*abs(cr[0]):.1f} {cr[1]/(2*abs(cr[0])):+.2f}")
print("\nPredicted argmax under this quadratic for U=1..10:", [int(torch.argmax(cg[1]*u*c + cr[0]*c**2 + cr[1]*c).item())+1 for u in range(1, 11)])

# per-count logit profiles (mean)
print("\nMean logits per true count (rows=U, cols=#1..#10):")
lg = r @ U.T
for u in range(1, 11):
    print(f"  U={u:2d}: " + " ".join(f"{v:7.1f}" for v in lg[counts == u].mean(0).tolist()))
