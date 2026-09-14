import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load("a")
print("params:", sum(p.numel() for p in m.parameters()))
rng = np.random.default_rng(0)
lists = rng.integers(0, 10, size=(4000, 5))
x = torch.tensor([tok1(l.tolist()) for l in lists])
logits, attn = m(x)
pred = logits[:, -1, :10].argmax(-1)
true = torch.tensor(lists.max(1))
print("accuracy (max):", (pred == true).float().mean().item())
# also: next token after the answer should be EOS
x2 = torch.cat([x, true[:, None]], 1)
logits2, _ = m(x2)
print("P(EOS after answer) argmax rate:", (logits2[:, -1].argmax(-1) == EOS).float().mean().item())

# Attention from ANS by head, averaged: how much goes to the max position(s)?
A = attn[0][:, :, -1, :]   # (B, 4, 11)
num_pos = torch.tensor([1, 3, 5, 7, 9])
vals = x[:, num_pos]        # (B, 5)
is_max = vals == true[:, None]
print("\nAttention from ANS, mean weight on: numbers that equal the max | other numbers | BOS | SEPs | ANS")
for h in range(4):
    a = A[:, h]
    on_max = (a[:, num_pos] * is_max).sum(-1).mean()
    on_other = (a[:, num_pos] * ~is_max).sum(-1).mean()
    print(f"  H{h}: {on_max:.3f} | {on_other:.3f} | {a[:, 0].mean():.3f} | {a[:, [2, 4, 6, 8]].sum(-1).mean():.3f} | {a[:, 10].mean():.3f}")

# Attention from ANS to a number token as a function of its value, holding others fixed: use the QK circuit directly
E = m.tok_embed.weight; P = m.pos_embed.weight
q_res = E[ANS] + P[10]
print("\nQK circuit: score from ANS (pos 10) to number token n at position p (p = 1,3,5,7,9 averaged), by head")
for h in range(4):
    hd = m.layers[0].heads[h]
    q = hd.W_Q(q_res)
    rows = []
    for n in range(10):
        ks = torch.stack([hd.W_K(E[n] + P[p]) for p in [1, 3, 5, 7, 9]])
        rows.append((ks @ q / 4).tolist())
    rows = np.array(rows)
    print(f"  H{h}: mean over positions: " + " ".join(f"{n}:{rows[n].mean():6.2f}" for n in range(10)) + f"   | position spread (std across p, avg over n): {rows.std(1).mean():.2f}")
    ksep = torch.stack([hd.W_K(E[SEP] + P[p]) for p in [2, 4, 6, 8]]) @ q / 4
    print(f"       SEP keys: {ksep.numpy().round(2)}   BOS key: {float(hd.W_K(E[BOS] + P[0]) @ q / 4):.2f}   ANS self key: {float(hd.W_K(E[ANS] + P[10]) @ q / 4):.2f}")
