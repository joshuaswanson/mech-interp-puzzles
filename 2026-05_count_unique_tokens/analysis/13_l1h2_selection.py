import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight

def layer0(x):
    b, s = x.shape
    h = E[x]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    outs = [head(h, mask)[0] for head in model.layers[0].heads]
    return h + model.layers[0].W_O(torch.cat(outs, -1))

rng = np.random.default_rng(9)
x, counts = random_batch(rng, 300)
B = len(counts)
resid = layer0(x)
seqs = x[:, 1:-1]
head = model.layers[1].heads[2]
q = resid[:, -1] @ head.W_Q.weight.T
k = resid @ head.W_K.weight.T
scores = torch.einsum("bd,bjd->bj", q, k) / D_HEAD ** 0.5   # (B, 12)
sym_scores = scores[:, 1:-1]

prefix_pres = torch.zeros(B, SEQ_LEN, 10)
for b in range(B):
    seen = torch.zeros(10)
    for j in range(SEQ_LEN):
        seen[seqs[b, j]] = 1
        prefix_pres[b, j] = seen
full = prefix_pres[:, -1]

# Regress score on prefix presence (10) + query-token identity one-hot (10)
onehot_q = torch.nn.functional.one_hot(seqs, 10).float()
X = torch.cat([prefix_pres, onehot_q], -1).reshape(-1, 20)
y = sym_scores.reshape(-1)
X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
w = torch.linalg.lstsq(X1, y.unsqueeze(1)).solution.squeeze()
pred = X1 @ w
print(f"L1H2 score at symbol pos ~ prefix presence + own token:  R^2 = {1 - ((pred - y)**2).mean() / y.var():.3f}")
print("  prefix-presence coeffs: " + " ".join(f"{chr(97+i)}={w[i]:6.1f}" for i in range(10)))
print("  own-token coeffs      : " + " ".join(f"{chr(97+i)}={w[10+i]:6.1f}" for i in range(10)))

# Within-sequence: score of complete-prefix positions vs incomplete
adgh = [0, 3, 6, 7]
n_full = full[:, adgh].sum(-1)
complete = prefix_pres[:, :, adgh].sum(-1) == n_full.unsqueeze(1)
not_bf = (seqs != 1) & (seqs != 5)
print(f"\nMean L1H2 score: complete-prefix & not b/f: {sym_scores[complete & not_bf].mean():.1f}   incomplete & not b/f: {sym_scores[~complete & not_bf].mean():.1f}   b positions: {sym_scores[seqs==1].mean():.1f}   f positions: {sym_scores[seqs==5].mean():.1f}")
print(f"BOS score mean: {scores[:,0].mean():.1f}   ANS score mean: {scores[:,-1].mean():.1f}")

# argmax position analysis
am = sym_scores.argmax(-1)
am_complete = complete[torch.arange(B), am]
print(f"\nargmax symbol position has complete {{a,d,g,h}} prefix: {am_complete.float().mean():.3f}")
print(f"argmax is position index (1-10) distribution: {np.bincount((am+1).numpy(), minlength=11)[1:]}")
# how much of the time is the argmax at the *first* complete-prefix position vs later?
first_complete = complete.float().argmax(-1)
print(f"argmax == first complete-prefix position: {(am == first_complete).float().mean():.3f}")

# Sequences with no a,d,g,h at all: where does it attend?
none = n_full == 0
print(f"\nSequences with no a/d/g/h (n={none.sum().item()}): argmax position dist: {np.bincount((am[none]+1).numpy(), minlength=11)[1:]}")
