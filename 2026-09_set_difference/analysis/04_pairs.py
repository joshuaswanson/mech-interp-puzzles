import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

m = load_model()
E, P, U, L = m.tok_embed.weight, m.pos_embed.weight, m.unembed.weight, m.layers[0]
ORDER = "dahgbkefionplcjm"
r9 = E[SEP] + P[9]
PX, PY = P[1], P[6]

print("Per symbol t and head h: unnormalised attention weight e^score and value v, at an X position and at a Y position.")
print("Paired symbols (in both X and Y) contribute e_X v_X + e_Y v_Y with total weight e_X + e_Y; ratio = their 'effective value'.")
for h in range(2):
    hd = L.heads[h]
    q = float(hd.W_Q.weight[0] @ r9)
    wk, wv = hd.W_K.weight[0], hd.W_V.weight[0]
    print(f"\n=== head {h} ===")
    print(f"{'t':>2s} {'e_X':>7s} {'v_X':>7s} {'e_Y':>7s} {'v_Y':>7s} | {'e_X+e_Y':>8s} {'eff value pair':>14s} | {'v_X (z alone)':>13s}")
    for ch in ORDER:
        t = ord(ch) - 97
        eX = float(torch.exp(q * (wk @ (E[t] + PX)))); vX = float(wv @ (E[t] + PX))
        eY = float(torch.exp(q * (wk @ (E[t] + PY)))); vY = float(wv @ (E[t] + PY))
        print(f"{ch:>2s} {eX:7.3f} {vX:7.1f} {eY:7.3f} {vY:7.1f} | {eX+eY:8.3f} {(eX*vX+eY*vY)/(eX+eY):14.2f} | {vX:13.1f}")
    for name, tok, pos in [("BOS", BOS, 0), ("SEP5", SEP, 5), ("SEP9", SEP, 9)]:
        e = float(torch.exp(q * (wk @ (E[tok] + P[pos])))); v = float(wv @ (E[tok] + P[pos]))
        print(f"{name:>4s} e={e:7.3f} v={v:7.1f}  e*v={e*v:7.2f}")
