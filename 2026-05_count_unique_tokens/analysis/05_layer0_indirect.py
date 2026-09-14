import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight
U = model.unembed.weight[COUNT_BASE:]

def forward_with_ablation(x, ablate_l0_heads=(), ablate_l0_at=None):
    """ablate_l0_at: None -> all positions; 'symbols' -> only symbol positions; 'ans' -> only ANS."""
    b, s = x.shape
    h = E[x]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    for L, layer in enumerate(model.layers):
        head_outs = []
        for hi, head in enumerate(layer.heads):
            out, attn = head(h, mask)
            if L == 0 and hi in ablate_l0_heads:
                if ablate_l0_at is None:
                    out = torch.zeros_like(out)
                elif ablate_l0_at == "symbols":
                    out = out.clone(); out[:, 1:-1] = 0
                elif ablate_l0_at == "ans":
                    out = out.clone(); out[:, -1] = 0
            head_outs.append(out)
        h = h + layer.W_O(torch.cat(head_outs, -1))
    return h[:, -1] @ U.T

rng = np.random.default_rng(2)
x, counts = random_batch(rng, 200)
base = forward_with_ablation(x)
print("baseline acc:", (base.argmax(-1) + 1 == counts).float().mean().item())

for where in [None, "symbols", "ans"]:
    print(f"\nZero-ablating layer-0 head outputs at positions={where or 'all'}:")
    for hs in [(0,), (1,), (2,), (3,), (0, 1, 2, 3)]:
        lg = forward_with_ablation(x, hs, where)
        acc = (lg.argmax(-1) + 1 == counts).float().mean().item()
        print(f"  heads {hs}: acc = {acc:.3f}")
