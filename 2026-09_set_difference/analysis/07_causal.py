import sys, copy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load_model()
E, P, U, L = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight, m.layers[0]
prompts = []
for X in itertools.combinations(range(16), 4):
    for z in X:
        Y = [s for s in X if s != z]
        prompts.append((encode(X, Y), z))
x = torch.tensor([p for p, _ in prompts]); z = torch.tensor([zz for _, zz in prompts])
present = torch.zeros(len(x), 16, dtype=torch.bool)
for i, (p, _) in enumerate(prompts):
    for t in p[1:5]: present[i, t] = True

def acc_split(model, t):
    logits, _ = model(x)
    ok = logits[:, -1, :16].argmax(-1) == z
    is_z = z == t; is_comp = present[:, t] & ~is_z; absent = ~present[:, t]
    return ok[absent].float().mean().item(), ok[is_comp].float().mean().item(), ok[is_z].float().mean().item()

print("Perturb one symbol's embedding off the cancellation curve (scale it by 1.5). Accuracy split by that symbol's role:")
print(f"{'sym':>3s} {'absent':>7s} {'companion':>10s} {'missing':>8s}")
for ch in "dkm":
    t = ord(ch) - 97
    mm = copy.deepcopy(m)
    mm.tok_embed.weight[t] *= 1.5
    a, c, zz = acc_split(mm, t)
    print(f"{ch:>3s} {a:7.3f} {c:10.3f} {zz:8.3f}")

print("\nRemove the X/Y positional difference (give Y positions the X positional vector), so rho = 1 and delta = 0:")
mm = copy.deepcopy(m)
mm.pos_embed.weight[6:9] = mm.pos_embed.weight[1]
logits, _ = mm(x); print(f"  accuracy = {(logits[:, -1, :16].argmax(-1) == z).float().mean():.3f}")

print("\nZero the direct path (remove the final SEP's own embedding + position from the readout):")
r = E[x] + P[torch.arange(10)]
logits_full, attn = m(x)
s = torch.stack([(attn[0][:, h, -1] * (r @ L.heads[h].W_V.weight[0])).sum(-1) for h in range(2)], 1)
heads_only = s[:, :1] * L.W_O.weight[:, 0] + s[:, 1:] * L.W_O.weight[:, 1]
print(f"  accuracy with heads only = {((heads_only @ U[:16].T).argmax(-1) == z).float().mean():.3f}")

print("\nSwap the two heads' normalisers (use Z1 for head 0 and Z0 for head 1): tests that only the ratio matters")
Zs, Ns = [], []
for h in range(2):
    hd = L.heads[h]; q = float(hd.W_Q.weight[0] @ (E[SEP] + P[9]))
    e = torch.exp(q * (r @ hd.W_K.weight[0])); v = r @ hd.W_V.weight[0]
    Zs.append(e.sum(-1)); Ns.append((e * v).sum(-1))
for name, (d0, d1) in {"actual (N0/Z0, N1/Z1)": (Zs[0], Zs[1]), "both divided by Z0": (Zs[0], Zs[0]), "both divided by Z1": (Zs[1], Zs[1]), "no normalisation at all": (torch.ones_like(Zs[0]), torch.ones_like(Zs[1]))}.items():
    fin = (E[SEP] + P[9]) + (Ns[0] / d0)[:, None] * L.W_O.weight[:, 0] + (Ns[1] / d1)[:, None] * L.W_O.weight[:, 1]
    print(f"  {name:28s}: accuracy = {((fin @ U[:16].T).argmax(-1) == z).float().mean():.3f}")
