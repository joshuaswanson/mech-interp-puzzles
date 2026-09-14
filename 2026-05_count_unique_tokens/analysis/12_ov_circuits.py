import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model = load_model()
E = model.tok_embed.weight
U = model.unembed.weight[COUNT_BASE:]
idx = torch.arange(1, 11).float()
idx_c = idx - idx.mean()
def slope(v):
    return ((v @ U.T) * idx_c).sum(-1) / (idx_c ** 2).sum()

L0, L1 = model.layers
def ov0(h):  # layer-0 head h OV map: (32 -> 32)
    return L0.W_O.weight[:, h * D_HEAD:(h + 1) * D_HEAD] @ L0.heads[h].W_V.weight
def ov1(h):
    return L1.W_O.weight[:, h * D_HEAD:(h + 1) * D_HEAD] @ L1.heads[h].W_V.weight

syms = [chr(97 + k) for k in range(10)] + ["BOS", "ANS"]

print("=== L1H2 value path: count-slope of OV1[2] @ OV0[h] @ E[k]  (what L0 head h writes for symbol k, as read by L1H2) ===")
print("        " + " ".join(f"{s:>6s}" for s in syms))
for h in range(4):
    v = (ov1(2) @ ov0(h) @ E[:12].T).T
    print(f"  L0H{h}: " + " ".join(f"{s:6.1f}" for s in slope(v).tolist()))
print("  direct (OV1[2] @ E[k]): " + " ".join(f"{s:6.1f}" for s in slope((ov1(2) @ E[:12].T).T).tolist()))

print("\n=== L1H3 value path: count-slope of OV1[3] @ E[k] and OV1[3] @ OV0[h] @ E[k] ===")
print("        " + " ".join(f"{s:>6s}" for s in syms))
print("  direct: " + " ".join(f"{s:6.1f}" for s in slope((ov1(3) @ E[:12].T).T).tolist()))
for h in range(4):
    v = (ov1(3) @ ov0(h) @ E[:12].T).T
    print(f"  L0H{h}: " + " ".join(f"{s:6.1f}" for s in slope(v).tolist()))

print("\n=== L1H3 QK: how L0 head h writing symbol k at ANS changes the score on the BOS key ===")
# BOS key in layer 1 = W_K (E[BOS] + L0_out[BOS]); L0_out[BOS] = sum_h OV0[h] E[BOS]
bos_resid = E[BOS] + sum(ov0(h) @ E[BOS] for h in range(4))
kBOS = L1.heads[3].W_K.weight @ bos_resid
print("        " + " ".join(f"{s:>6s}" for s in syms))
for h in range(4):
    q = (L1.heads[3].W_Q.weight @ ov0(h) @ E[:12].T).T   # (12, 8)
    sc = (q @ kBOS) / D_HEAD ** 0.5
    print(f"  L0H{h}: " + " ".join(f"{s:6.2f}" for s in sc.tolist()))
q = (L1.heads[3].W_Q.weight @ E[:12].T).T
print("  direct: " + " ".join(f"{s:6.2f}" for s in ((q @ kBOS) / D_HEAD ** 0.5).tolist()))

print("\n=== L1H0 / L1H1 value path: count-slope of OV1[h] @ E[k] (direct) and via L0 heads ===")
for h1 in [0, 1]:
    print(f"  L1H{h1} direct: " + " ".join(f"{s:6.1f}" for s in slope((ov1(h1) @ E[:12].T).T).tolist()))
    for h in range(4):
        v = (ov1(h1) @ ov0(h) @ E[:12].T).T
        print(f"  L1H{h1} L0H{h} : " + " ".join(f"{s:6.1f}" for s in slope(v).tolist()))
