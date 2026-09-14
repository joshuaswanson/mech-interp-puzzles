import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load("b")
print("params:", sum(p.numel() for p in m.parameters()))
rng = np.random.default_rng(0)
lists = rng.integers(0, 100, size=(3000, 5))
x = torch.tensor([tok2(l.tolist()) for l in lists])
true = torch.tensor(lists.max(1)); T, O = true // 10, true % 10
logits, attn = m(x)
pred_T = logits[:, -1, :10].argmax(-1)
x2 = torch.cat([x, T[:, None]], 1)
logits2, attn2 = m(x2)
pred_O = logits2[:, -1, :10].argmax(-1)
x3 = torch.cat([x2, O[:, None]], 1)
logits3, _ = m(x3)
print("tens acc:", (pred_T == T).float().mean().item(), " ones acc:", (pred_O == O).float().mean().item(), " EOS acc:", (logits3[:, -1].argmax(-1) == EOS).float().mean().item())

labels = [label(t) for t in x2[0].tolist()]
TENS = [1, 4, 7, 10, 13]; ONES = [2, 5, 8, 11, 14]; SEPS = [3, 6, 9, 12]
tens_tok = x[:, TENS]; ones_tok = x[:, ONES]
nums = tens_tok * 10 + ones_tok
is_maxnum = nums == true[:, None]
is_maxtens = tens_tok == T[:, None]

def report(A, name, B):
    # A: (B, 4, seq) attention row; summarize mass by group
    print(f"\n{name}")
    for h in range(4):
        a = A[:, h]
        print(f"  L?H{h}: BOS {a[:,0].mean():.2f} | tens {a[:,TENS].sum(-1).mean():.2f} (of which max-number {(a[:,TENS]*is_maxnum).sum(-1).mean():.2f}, max-tens {(a[:,TENS]*is_maxtens).sum(-1).mean():.2f}) | ones {a[:,ONES].sum(-1).mean():.2f} (max-number {(a[:,ONES]*is_maxnum).sum(-1).mean():.2f}, max-tens {(a[:,ONES]*is_maxtens).sum(-1).mean():.2f}) | SEP {a[:,SEPS].sum(-1).mean():.2f} | ANS {a[:,15].mean():.2f}" + (f" | T {a[:,16].mean():.2f}" if a.shape[1] > 16 else ""))

report(attn[0][:, :, 15, :], "Layer 0, query = ANS (pos 15), predicting the tens digit", len(x))
report(attn[1][:, :, 15, :], "Layer 1, query = ANS (pos 15)", len(x))
report(attn2[0][:, :, 16, :], "Layer 0, query = fed-back tens digit (pos 16), predicting the ones digit", len(x))
report(attn2[1][:, :, 16, :], "Layer 1, query = fed-back tens digit (pos 16)", len(x))

print("\nLayer 0 attention FROM ones-digit positions (where would a number get composed?): mean weight on own tens digit (previous position), own position, BOS")
A0 = attn[0]
for h in range(4):
    own_tens = torch.stack([A0[:, h, p, p - 1] for p in ONES], 1).mean()
    self_w = torch.stack([A0[:, h, p, p] for p in ONES], 1).mean()
    bos = torch.stack([A0[:, h, p, 0] for p in ONES], 1).mean()
    print(f"  L0H{h}: own tens {own_tens:.2f}  self {self_w:.2f}  BOS {bos:.2f}")
print("Layer 0 attention FROM SEP positions: mean weight on the preceding ones digit, preceding tens digit, self, BOS")
for h in range(4):
    o = torch.stack([A0[:, h, p, p - 1] for p in SEPS], 1).mean(); t = torch.stack([A0[:, h, p, p - 2] for p in SEPS], 1).mean()
    s = torch.stack([A0[:, h, p, p] for p in SEPS], 1).mean(); b = torch.stack([A0[:, h, p, 0] for p in SEPS], 1).mean()
    print(f"  L0H{h}: ones {o:.2f}  tens {t:.2f}  self {s:.2f}  BOS {b:.2f}")
