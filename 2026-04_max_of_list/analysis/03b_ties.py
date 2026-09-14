import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load("b")
E, P, U = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight
L0, L1 = m.layers
D = 16
ONES = [2, 5, 8, 11, 14]

def resid_after_l0(x):
    b, s = x.shape
    h = E[x] + P[torch.arange(s)]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    parts = {}
    outs = []
    for hi, head in enumerate(L0.heads):
        o, _ = head(h, mask); outs.append(o)
        parts[f"L0H{hi}"] = o @ L0.W_O.weight[:, hi * D:(hi + 1) * D].T
    parts["tok+pos"] = h
    return h + L0.W_O(torch.cat(outs, -1)), parts

print("Constructed lists where several numbers share the max tens digit. Attention of L1H0/L1H3 from the output position over the five ones-digit positions, and the prediction.")
cases = [[83, 87, 12, 45, 60], [87, 83, 12, 45, 60], [90, 99, 95, 91, 93], [50, 51, 52, 53, 54], [39, 31, 30, 35, 12], [64, 69, 61, 8, 3], [20, 20, 20, 20, 21]]
for nums in cases:
    T = max(nums) // 10; O = max(nums) % 10
    x = torch.tensor([tok2(nums) + [T]])
    logits, attn = m(x)
    pred = int(logits[0, -1, :10].argmax())
    a0 = attn[1][0, 0, 16, ONES]; a3 = attn[1][0, 3, 16, ONES]
    print(f"  {str(nums):24s} max {max(nums)}  pred ones {pred}  {'ok' if pred == O else 'WRONG'}   L1H0 over ones positions: " + " ".join(f"{n%10}:{w:.2f}" for n, w in zip(nums, a0.tolist())) + "   L1H3: " + " ".join(f"{n%10}:{w:.2f}" for n, w in zip(nums, a3.tolist())))

print("\nScore decomposition for L1H0 at the output position: which parts of the key (residual after layer 0 at each ones position) carry the score?")
nums = [83, 87, 12, 45, 60]; T = 8
x = torch.tensor([tok2(nums) + [T]])
r1, parts = resid_after_l0(x)
hd = L1.heads[0]
q = hd.W_Q(r1[0, 16])
print(f"  list {nums}, query = residual at pos 16 (token {T})")
print(f"  {'part':8s} " + " ".join(f"{'num '+str(n):>8s}" for n in nums))
for name in ["tok+pos", "L0H0", "L0H1", "L0H2", "L0H3"]:
    row = [float(hd.W_K(parts[name][0, p]) @ q / 4) for p in ONES]
    print(f"  {name:8s} " + " ".join(f"{v:8.2f}" for v in row))
row = [float(hd.W_K(r1[0, p]) @ q / 4) for p in ONES]
print(f"  {'total':8s} " + " ".join(f"{v:8.2f}" for v in row))

print("\nEmpirical: over random lists whose top two numbers share the max tens digit, how often does L1H0 put more weight on the larger ones digit, and ones accuracy")
rng = np.random.default_rng(4)
n_ok = n_tot = n_att = 0
for _ in range(2000):
    t = int(rng.integers(1, 10)); o1, o2 = rng.choice(10, size=2, replace=False)
    others = rng.integers(0, t * 10, size=3)
    nums = [t * 10 + int(o1), t * 10 + int(o2)] + others.tolist(); rng.shuffle(nums)
    T = max(nums) // 10; O = max(nums) % 10
    x = torch.tensor([tok2(nums) + [T]]); logits, attn = m(x)
    n_tot += 1; n_ok += int(logits[0, -1, :10].argmax()) == O
    w = attn[1][0, 0, 16, ONES]; big = [i for i, n in enumerate(nums) if n == max(nums)][0]
    n_att += int(w.argmax()) == big
print(f"  ones accuracy {n_ok/n_tot:.3f};  L1H0 top attention on the max number's ones digit {n_att/n_tot:.3f}")
