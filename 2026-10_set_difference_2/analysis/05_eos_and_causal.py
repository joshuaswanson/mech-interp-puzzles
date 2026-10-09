import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

model, config, mod = load_model()
E, U = model.tok_embed.weight, model.unembed.weight
Uf = U * model.ln_final.weight

def rms(v): return (v.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()

def run(tokens, l1_override=None, l0_override=None):
    # l0_override / l1_override: dict position -> attention row to substitute
    x = torch.tensor([tokens]); n = len(tokens)
    rope = mod.rope_cos_sin(n, 64, 10000.0, x.device)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    r0 = E[x]
    def attend(blk, r, override):
        h = blk.ln_attn(r)
        q, k, v = blk.attn.W_Q(h), blk.attn.W_K(h), blk.attn.W_V(h)
        cos, sin = rope
        q = q * cos + mod.rotate_half(q) * sin
        k = k * cos + mod.rotate_half(k) * sin
        sc = (q @ k.transpose(-1, -2)) / 8.0
        sc = sc.masked_fill(~mask, float("-inf"))
        p = sc.softmax(-1)
        if override:
            p = p.clone()
            for pos, row in override.items(): p[0, pos] = row
        return blk.attn.W_O(p @ v), p
    a0, p0 = attend(model.blocks[0], r0, l0_override)
    r1 = r0 + a0
    a1, p1 = attend(model.blocks[1], r1, l1_override)
    r2 = r1 + a1
    return dict(logits=model.unembed(model.ln_final(r2))[0], embed=r0[0], L0=a0[0], L1=a1[0], final=r2[0], p0=p0[0], p1=p1[0], r1=r1[0])

print("1. EOS logit decomposed by component, at the final position (all consumed) vs the step before")
rng = random.Random(5)
for K in [3, 5, 8]:
    acc = {"last": [], "prev": []}
    for _ in range(100):
        toks = make_prompt(rng, K); t = run(toks)
        for name, i in [("last", 2 * K + 1), ("prev", 2 * K)]:
            sc = float(rms(t["final"][i]))
            acc[name].append([float(Uf[EOS] @ t[p][i]) / sc for p in ("embed", "L0", "L1")] + [float(t["logits"][i, EOS])])
    for name in ("last", "prev"):
        a = np.array(acc[name]).mean(0)
        print(f"  K={K} {name:4s}: embed {a[0]:7.2f}  L0 {a[1]:7.2f}  L1 {a[2]:7.2f}  total {a[3]:7.2f}")

print("\n2. Force layer 1 to keep attending to SEP at the final position. Does EOS still fire?")
for K in [3, 5, 8]:
    base, forced = [], []
    for _ in range(100):
        toks = make_prompt(rng, K); i = 2 * K + 1
        base.append(float(run(toks)["logits"][i, EOS]))
        row = torch.zeros(len(toks)); row[K + 1] = 1.0
        forced.append(float(run(toks, l1_override={i: row})["logits"][i, EOS]))
    print(f"  K={K}: EOS logit {np.mean(base):7.2f} -> {np.mean(forced):7.2f} when layer 1 is pinned to SEP")

print("\n3. What does layer 1 attend to at the final position instead of SEP?")
for K in [3, 5, 8]:
    acc = np.zeros(4)
    for _ in range(100):
        toks = make_prompt(rng, K); t = run(toks); i = 2 * K + 1
        p = t["p1"][i]
        acc += np.array([float(p[0]), float(p[1:K + 1].sum()), float(p[K + 1]), float(p[K + 2:2 * K + 2].sum())])
    acc /= 100
    print(f"  K={K}: BOS {acc[0]:.3f}  X positions {acc[1]:.3f}  SEP {acc[2]:.3f}  Y positions {acc[3]:.3f}")

print("\n4. Causal: break the count signal by making layer 0 attend only to the X block (ignore consumed Y symbols).")
print("   If the count is what tracks consumption, every X symbol should look valid again.")
for K in [4, 6]:
    agree = []
    for _ in range(60):
        toks = make_prompt(rng, K); Xs = set(toks[1:K + 1])
        for i in range(K + 2, 2 * K + 1):
            row = torch.zeros(len(toks)); row[1:K + 1] = 1.0 / K
            lg = run(toks, l0_override={i: row})["logits"][i]
            top = set(int(s) for s in lg[:NUM_SYMBOLS].topk(K).indices)
            agree.append(top == Xs)
    print(f"  K={K}: top-K symbols equal the whole of X in {np.mean(agree):.3f} of cases (so consumption is forgotten)")

print("\n5. Causal: hand-edit the count. Insert an extra copy of one not-yet-consumed symbol into layer 0's average")
print("   by giving its X position double weight. Its logit should fall to the consumed level.")
for K in [4, 6]:
    drops = []
    for _ in range(60):
        toks = make_prompt(rng, K); i = K + 2
        Xs = toks[1:K + 1]; consumed = set(toks[K + 2:i + 1])
        cand = [(j + 1, s) for j, s in enumerate(Xs) if s not in consumed]
        if not cand: continue
        pos, s = cand[0]
        t = run(toks); base = float(t["logits"][i, s])
        row = t["p0"][i].clone(); row[pos] *= 2; row = row / row.sum()
        new = float(run(toks, l0_override={i: row})["logits"][i, s])
        others = [float(t["logits"][i, ss]) for ss in Xs if ss in consumed]
        drops.append((base, new, np.mean(others) if others else np.nan))
    d = np.array(drops)
    print(f"  K={K}: logit of that symbol {d[:,0].mean():6.2f} -> {d[:,1].mean():6.2f};  genuinely consumed symbols sit at {np.nanmean(d[:,2]):6.2f}")
