import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight
U = model.unembed.weight[COUNT_BASE:]

def run_decomposed(x):
    """Return per-component contributions to the ANS residual: dict name -> (B, d_model)."""
    b, s = x.shape
    h = E[x]
    mask = torch.tril(torch.ones(s, s)).unsqueeze(0)
    comps = {"embed": h[:, -1].clone()}
    attn_all = []
    for L, layer in enumerate(model.layers):
        head_outs = []
        attns = []
        for hi, head in enumerate(layer.heads):
            out, attn = head(h, mask)
            head_outs.append(out)
            attns.append(attn)
        attn_all.append(torch.stack(attns, 1))
        W_O = layer.W_O.weight  # (d_model, d_model)
        total = torch.zeros_like(h)
        for hi, ho in enumerate(head_outs):
            contrib = ho @ W_O[:, hi * D_HEAD:(hi + 1) * D_HEAD].T
            comps[f"L{L}H{hi}"] = contrib[:, -1].clone()
            total = total + contrib
        h = h + total
    return comps, h[:, -1], attn_all

rng = np.random.default_rng(1)
x, counts = random_batch(rng, 200)
comps, final, attn_all = run_decomposed(x)
logits_check = final @ U.T
assert (logits_check.argmax(-1) + 1 == counts).all()

print("Mean contribution of each component to the CORRECT count logit, minus mean over other count logits (logit diff):")
print(f"{'component':10s} " + " ".join(f"c={c:2d}" for c in range(1, 11)) + "   avg")
for name, v in comps.items():
    lg = v @ U.T  # (B, 10)
    correct = lg[torch.arange(len(counts)), counts - 1]
    others = (lg.sum(-1) - correct) / 9
    diff = correct - others
    per_c = [diff[counts == c].mean().item() for c in range(1, 11)]
    print(f"{name:10s} " + " ".join(f"{d:5.2f}" for d in per_c) + f"  {diff.mean().item():5.2f}")

print("\nMean-ablation of each component (replace by mean over batch) -> accuracy:")
for name in comps:
    alt = final - comps[name] + comps[name].mean(0, keepdim=True)
    acc = ((alt @ U.T).argmax(-1) + 1 == counts).float().mean().item()
    print(f"  ablate {name:8s}: acc = {acc:.3f}")

print("\nZero-ablation of each component -> accuracy:")
for name in comps:
    alt = final - comps[name]
    acc = ((alt @ U.T).argmax(-1) + 1 == counts).float().mean().item()
    print(f"  ablate {name:8s}: acc = {acc:.3f}")

print("\nKeep ONLY a subset of components (others zeroed) -> accuracy:")
for keep in [["L1H0","L1H1","L1H2","L1H3"], ["L0H0","L0H1","L0H2","L0H3"], ["embed","L1H0","L1H1","L1H2","L1H3"], ["embed","L0H0","L0H1","L0H2","L0H3"]]:
    alt = sum(comps[k] for k in keep)
    acc = ((alt @ U.T).argmax(-1) + 1 == counts).float().mean().item()
    print(f"  keep {keep}: acc = {acc:.3f}")
