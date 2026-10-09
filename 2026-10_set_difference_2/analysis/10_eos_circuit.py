import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()
E = model.tok_embed.weight
b0, b1 = model.blocks[0], model.blocks[1]
rng = random.Random(51)

def pieces(tokens):
    x = torch.tensor([tokens]); n = len(tokens)
    rope = mod.rope_cos_sin(n, 64, 10000.0, x.device)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    r0 = E[x]
    a0, p0 = b0.attn(b0.ln_attn(r0), mask, rope)
    r1 = r0 + a0
    h = b1.ln_attn(r1)
    q, k, v = b1.attn.W_Q(h), b1.attn.W_K(h), b1.attn.W_V(h)
    cos, sin = rope
    qr = q * cos + mod.rotate_half(q) * sin
    kr = k * cos + mod.rotate_half(k) * sin
    S = (qr[0] @ kr[0].T) / 8.0
    return r1, v[0], S, rope

def logits_with_forced_l1(tokens, pos, key):
    r1, v, _, _ = pieces(tokens)
    row = torch.zeros(len(tokens)); row[key] = 1.0
    a1 = b1.attn.W_O(row @ v)
    r2 = r1[0].clone(); r2[pos] = r1[0, pos] + a1
    return model.unembed(model.ln_final(r2))[pos]

print("1. Force layer 1 at the final position onto each key in turn. Which key carries EOS?\n")
for K in [4, 6]:
    acc = {}
    for _ in range(50):
        toks = make_prompt(rng, K); i = 2 * K + 1
        for key in range(i + 1):
            role = "BOS" if key == 0 else ("SEP" if key == K + 1 else (f"x{key}" if key <= K else f"y{key - K - 1}"))
            acc.setdefault((key, role), []).append(float(logits_with_forced_l1(toks, i, key)[EOS]))
    print(f"  K={K}: EOS logit when layer 1 is pinned to each key")
    for (key, role), v in sorted(acc.items()):
        bar = "#" * max(0, int(np.mean(v) / 4))
        print(f"    key {key:2d} ({role:4s}): {np.mean(v):8.2f} {bar}")
    print()

print("2. Attention actually paid at the final position, by key role.\n")
for K in [4, 6, 8]:
    acc = {}
    for _ in range(80):
        toks = make_prompt(rng, K); i = 2 * K + 1
        _, attns = model(torch.tensor([toks]))
        p = attns[1][0, 0, i]
        for key in range(i + 1):
            role = "BOS" if key == 0 else ("SEP" if key == K + 1 else (f"x{key}" if key <= K else f"y{key - K - 1}"))
            acc.setdefault((key, role), []).append(float(p[key]))
    top = sorted(acc.items(), key=lambda kv: -np.mean(kv[1]))[:4]
    print(f"  K={K}: " + "   ".join(f"{role}@{key}: {np.mean(v):.3f}" for (key, role), v in top))

print("\n3. Why does y1's key become competitive only at the end? Score on the y1 key and on SEP,")
print("   as the query moves away from SEP.\n")
for K in [4, 6, 8]:
    rows = {}
    for _ in range(60):
        toks = make_prompt(rng, K)
        _, _, S, _ = pieces(toks)
        for i in range(K + 1, 2 * K + 2):
            rows.setdefault(i - (K + 1), []).append((float(S[i, K + 1]), float(S[i, K + 2])))
    print(f"  K={K}")
    for d, v in sorted(rows.items()):
        if d == 0: continue
        a = np.array(v)
        print(f"    d={d}: SEP {a[:,0].mean():7.2f}   y1 key {a[:,1].mean():7.2f}   SEP minus y1 {a[:,0].mean()-a[:,1].mean():7.2f}")

print("\n4. RoPE sweep for the y1 key with content fixed, to separate rotation from content.\n")
K = 6
toks = make_prompt(rng, K)
r1, v, _, _ = pieces(toks)
h = b1.ln_attn(r1)[0]
qv = b1.attn.W_Q(h[2 * K + 1])
print(f"  {'separation':>10s} {'SEP key':>9s} {'y1 key':>9s}")
for d in range(1, 11):
    cos, sin = mod.rope_cos_sin(20, 64, 10000.0, torch.device("cpu"))
    qq = qv * cos[d] + mod.rotate_half(qv) * sin[d]
    out = []
    for kpos in (K + 1, K + 2):
        kv = b1.attn.W_K(h[kpos])
        kk = kv * cos[0] + mod.rotate_half(kv) * sin[0]
        out.append(float(qq @ kk) / 8.0)
    print(f"  {d:10d} {out[0]:9.2f} {out[1]:9.2f}")

print("\n5. Does the EOS value live in y1 specifically, or in any y position?")
print("   Pin layer 1 to each y position at the final step and read the EOS logit.\n")
for K in [6]:
    acc = {}
    for _ in range(60):
        toks = make_prompt(rng, K); i = 2 * K + 1
        for j in range(K):
            acc.setdefault(j, []).append(float(logits_with_forced_l1(toks, i, K + 2 + j)[EOS]))
    print(f"  K={K}: " + "  ".join(f"y{j}: {np.mean(v):7.2f}" for j, v in sorted(acc.items())))
