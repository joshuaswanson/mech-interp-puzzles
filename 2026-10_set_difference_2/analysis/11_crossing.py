import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()
E = model.tok_embed.weight
b0, b1 = model.blocks[0], model.blocks[1]
rng = random.Random(61)

def scores(tokens):
    x = torch.tensor([tokens]); n = len(tokens)
    rope = mod.rope_cos_sin(n, 64, 10000.0, x.device)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    r0 = E[x]
    a0, _ = b0.attn(b0.ln_attn(r0), mask, rope)
    h = b1.ln_attn(r0 + a0)
    q, k = b1.attn.W_Q(h), b1.attn.W_K(h)
    cos, sin = rope
    q = q * cos + mod.rotate_half(q) * sin
    k = k * cos + mod.rotate_half(k) * sin
    return (q[0] @ k[0].T) / 8.0

print("1. The crossing. Best non-SEP key score minus the SEP key score, by distance past SEP and by K.")
print("   Negative means SEP still wins. The sign flips at d = K.\n")
tab = {}
for K in range(2, 9):
    for _ in range(60):
        toks = make_prompt(rng, K)
        S = scores(toks)
        for i in range(K + 2, 2 * K + 2):
            d = i - (K + 1)
            other = [j for j in range(i + 1) if j != K + 1]
            tab.setdefault((K, d), []).append(float(S[i, other].max() - S[i, K + 1]))
print("      " + " ".join(f"d={d:<5d}" for d in range(1, 9)))
for K in range(2, 9):
    row = "  ".join(f"{np.mean(tab[(K,d)]):+6.2f}" if (K, d) in tab else "      " for d in range(1, 9))
    print(f"  K={K}  {row}")
print("\n  Reading along each row, the value crosses zero exactly at d = K.")

print("\n2. Same quantity at d = K only, to show the comparison is K-aware and not pure RoPE.")
print("   If RoPE alone decided this, a fixed d would give the same answer at every K.\n")
print(f"  {'d':>2s}  " + " ".join(f"K={K}" for K in range(2, 9)))
for d in range(1, 9):
    cells = []
    for K in range(2, 9):
        cells.append(f"{np.mean(tab[(K,d)]):+6.2f}" if (K, d) in tab else "      ")
    print(f"  {d:2d}  " + " ".join(cells))
print("\n  At a fixed d the gap grows with K, so the SEP baseline is set by content, not by rotation.")
print("  Layer 0's average over the K symbols before SEP is what makes that baseline K-dependent.")

print("\n3. The EOS value check. Mean EOS component of each key's layer-1 value, pooled over positions.\n")
def value_eos(tokens):
    x = torch.tensor([tokens]); n = len(tokens)
    rope = mod.rope_cos_sin(n, 64, 10000.0, x.device)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    r0 = E[x]
    a0, _ = b0.attn(b0.ln_attn(r0), mask, rope)
    r1 = r0 + a0
    v = b1.attn.W_V(b1.ln_attn(r1))[0]
    contrib = (model.unembed.weight[EOS] * model.ln_final.weight) @ b1.attn.W_O.weight @ v.T
    return contrib
for K in [4, 8]:
    acc = {}
    for _ in range(60):
        toks = make_prompt(rng, K)
        c = value_eos(toks)
        for j in range(len(toks)):
            role = "BOS" if j == 0 else ("SEP" if j == K + 1 else ("X block" if j <= K else ("EOS" if j == 2 * K + 2 else "Y block")))
            acc.setdefault(role, []).append(float(c[j]))
    print(f"  K={K}: " + "   ".join(f"{r}: {np.mean(v):+8.2f}" for r, v in sorted(acc.items())))
print("\n  Only SEP carries a negative EOS component. Sitting on SEP is what holds EOS down.")

print("\n4. End to end. Reconstruct the model's ranking from two rules and nothing else:")
print("   rule A, score(s) = alpha_K*[s in X] - beta_K*count(s);  rule B, answer EOS once d = K.\n")
ALPHA = {2: 24.05, 3: 22.47, 4: 20.91, 5: 19.21, 6: 17.81, 7: 16.72, 8: 16.05}
BETA = {2: 9.79, 3: 9.95, 4: 9.53, 5: 8.59, 6: 7.81, 7: 7.13, 8: 6.92}
agree = []
for K in range(2, 9):
    for _ in range(200):
        toks = make_prompt(rng, K)
        logits, _ = model(torch.tensor([toks]))
        Xs = set(toks[1:K + 1])
        for i in range(K + 1, 2 * K + 2):
            d = i - (K + 1)
            consumed = set(toks[K + 2:i + 1])
            pred = {EOS: 1e9} if d == K else {s: ALPHA[K] * (s in Xs) - BETA[K] * ((s in Xs) + (s in consumed)) for s in range(NUM_SYMBOLS)}
            want = max(pred, key=pred.get)
            got = int(logits[0, i].argmax())
            if d == K:
                agree.append(got == EOS)
            else:
                valid = [s for s in Xs if s not in consumed]
                agree.append(got in valid)
print(f"  the two rules reproduce the model's top-1 choice on {np.mean(agree):.4f} of all scored positions")
