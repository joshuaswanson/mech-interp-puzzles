import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()
E = model.tok_embed.weight
b0, b1 = model.blocks[0], model.blocks[1]
rng = random.Random(41)

def resid_after_l0(tokens):
    x = torch.tensor([tokens]); n = len(tokens)
    rope = mod.rope_cos_sin(n, 64, 10000.0, x.device)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    r0 = E[x]
    a0, p0 = b0.attn(b0.ln_attn(r0), mask, rope)
    return r0 + a0, p0, rope

def l1_scores(tokens):
    """Raw layer-1 attention scores from every query position, with RoPE applied."""
    r1, _, rope = resid_after_l0(tokens)
    h = b1.ln_attn(r1)
    q, k = b1.attn.W_Q(h), b1.attn.W_K(h)
    cos, sin = rope
    q = q * cos + mod.rotate_half(q) * sin
    k = k * cos + mod.rotate_half(k) * sin
    return (q[0] @ k[0].T) / 8.0, q[0], k[0]

print("1. Layer-1 scores from the last scored position, by key, for one sequence of each K.")
print("   SEP is the key the head normally sits on.\n")
for K in [4, 6]:
    toks = make_prompt(rng, K)
    S, _, _ = l1_scores(toks)
    labs = [label(t) for t in toks]
    for i in [K + 2, 2 * K, 2 * K + 1]:
        row = " ".join(f"{labs[j]}{j}:{float(S[i, j]):6.1f}" for j in range(i + 1))
        tag = "last" if i == 2 * K + 1 else ("second to last" if i == 2 * K else "early")
        print(f"  K={K} query {i:2d} ({tag:14s}): {row}")
    print()

print("2. Is the SEP key special by content or by position? Score on the SEP key against")
print("   the mean score on the symbol keys, as the query moves away from SEP.\n")
for K in [4, 6, 8]:
    rows = {}
    for _ in range(60):
        toks = make_prompt(rng, K)
        S, _, _ = l1_scores(toks)
        for i in range(K + 1, 2 * K + 2):
            d = i - (K + 1)
            sym_keys = list(range(1, K + 1)) + list(range(K + 2, i + 1))
            rows.setdefault(d, []).append((float(S[i, K + 1]), float(S[i, sym_keys].mean()) if sym_keys else np.nan))
    print(f"  K={K}")
    for d, v in sorted(rows.items()):
        a = np.array(v)
        print(f"    d={d}: SEP key {a[:,0].mean():7.2f}   symbol keys {np.nanmean(a[:,1]):7.2f}   gap {a[:,0].mean()-np.nanmean(a[:,1]):7.2f}")

print("\n3. RoPE isolated. Take one fixed pair of residual vectors (the query at a y position and")
print("   the key at SEP) and sweep their separation, keeping content fixed.\n")
K = 6
toks = make_prompt(rng, K)
r1, _, _ = resid_after_l0(toks)
h = b1.ln_attn(r1)[0]
qv, kv = b1.attn.W_Q(h[2 * K]), b1.attn.W_K(h[K + 1])
print(f"  {'separation':>10s} {'score':>8s}")
for d in range(0, 12):
    n = 20
    cos, sin = mod.rope_cos_sin(n, 64, 10000.0, torch.device("cpu"))
    qq = qv * cos[d] + mod.rotate_half(qv) * sin[d]
    kk = kv * cos[0] + mod.rotate_half(kv) * sin[0]
    print(f"  {d:10d} {float(qq @ kk) / 8.0:8.2f}")
print("  Content held fixed, so any variation here is RoPE alone.")

print("\n4. Where does the layer-1 query at a y position come from? Replace the query's input")
print("   residual with one built from a different number of preceding symbols.\n")
print("   Cross-splice: run sequence A, but compute the layer-1 query at the last position")
print("   from sequence B's residual at the same absolute position, with B having a different K.")
for KA, KB in [(6, 4), (4, 6)]:
    out = []
    for _ in range(50):
        tA = make_prompt(rng, KA)
        # B shares the absolute position but has a different number of symbols before SEP
        XB = rng.sample(range(NUM_SYMBOLS), KB); YB = XB[:]; rng.shuffle(YB)
        pad = [XB[rng.randrange(KB)] for _ in range(2 * KA - 2 * KB)]
        tB = [BOS] + XB + [SEP] + YB + pad + [EOS]
        if len(tB) != len(tA): continue
        i = 2 * KA + 1
        rA, _, rope = resid_after_l0(tA)
        rB, _, _ = resid_after_l0(tB)
        hA, hB = b1.ln_attn(rA), b1.ln_attn(rB)
        q = b1.attn.W_Q(hB)[:, i:i + 1]; k = b1.attn.W_K(hA)
        cos, sin = rope
        q = q * cos[i] + mod.rotate_half(q) * sin[i]
        k = k * cos + mod.rotate_half(k) * sin
        s = (q[0, 0] @ k[0].T) / 8.0
        s = s.masked_fill(torch.arange(len(tA)) > i, float("-inf")).softmax(-1)
        out.append(float(s[KA + 1]))
    if out:
        print(f"  query from a K={KB} sequence used in a K={KA} sequence at its last position: SEP mass {np.mean(out):.3f}")
for K in [4, 6]:
    m = []
    for _ in range(50):
        toks = make_prompt(rng, K)
        _, attns = model(torch.tensor([toks]))
        m.append(float(attns[1][0, 0, 2 * K + 1, K + 1]))
    print(f"  unspliced control, K={K} at its last position: SEP mass {np.mean(m):.3f}")
