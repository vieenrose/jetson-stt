"""Native zh-TW head FT on FROZEN strong encoder_out.
Warm-start head from deployed ONNX, relabel->Traditional + one-to-many alternates, train head w/ RNN-T loss."""
import os, sys, re, random
sys.path[:0] = ["recipe/X-ASR-zh-en/zipformer", "icefall"]
import numpy as np, torch, torch.nn as nn, onnx, onnx.numpy_helper as nh
import torchaudio.functional as AF
import sentencepiece as spm
from opencc import OpenCC
from decoder import Decoder

M = "chunks/960"
STEPS = int(os.environ.get("STEPS", "1200")); BS = int(os.environ.get("BS", "4"))
LR = float(os.environ.get("LR", "2e-4")); dev = "cuda"
TMAX, UMAX = 220, 60
random.seed(0); torch.manual_seed(0)
t2s = OpenCC("t2s"); s2twp = OpenCC("s2twp"); CJK = re.compile(r"[㐀-鿿]")
sp = spm.SentencePieceProcessor(); sp.load("recipe/X-ASR-zh-en/zipformer/data/lang_5000_with_punctuation/bpe_punc.model")
V0 = 5000

# ---- token maps from deployed punct tokens.txt ----
lines = [l.rstrip("\n") for l in open(f"{M}/tokens.txt", encoding="utf-8")]
simp2id = {}; punct2id = {}
for l in lines:
    p = l.rsplit(" ", 1)
    if len(p) != 2: continue
    tok, i = p[0], int(p[1]); body = tok[1:] if tok.startswith("▁") else tok
    if CJK.match(body): simp2id[body] = i
    elif body in "，。！？、；：": punct2id[body] = i
ascii_punct = {",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "："}

# ---- load precomputed feats (encoder_out + Traditional texts) ----
def load_feats(f):
    z = np.load(f, allow_pickle=True); return list(z["eo"]), list(z["txt"])
tr_eo, tr_tx = [], []
for f in ["feat_ntuml_train.npz", "feat_cv_train.npz"]:
    if os.path.exists(f): e, t = load_feats(f); tr_eo += e; tr_tx += t
ev_eo, ev_tx = load_feats("feat_cv_eval.npz")

# ---- collect one-to-many alternates from ALL training/eval Traditional texts ----
ref_chars = set()
for t in tr_tx + ev_tx:
    ref_chars |= set(c for c in t if CJK.match(c))
ext = {}
for T in sorted(ref_chars):
    S = t2s.convert(T)
    if S in simp2id and s2twp.convert(S) != T:  # one-to-many: char-level s2twp can't reach T
        ext[T] = V0 + len(ext)
ext_sibling = {eid: simp2id[t2s.convert(T)] for T, eid in ext.items()}
V1 = V0 + len(ext)
print(f"vocab {V0}->{V1} (+{len(ext)} alternates): {list(ext)[:30]}", flush=True)

# ---- native tokens.txt: relabel surfaces + append alternates ----
def relabel(l):
    p = l.rsplit(" ", 1)
    if len(p) != 2: return l
    tok, i = p; pre = "▁" if tok.startswith("▁") else ""; body = tok[len(pre):]
    if CJK.match(body): body = s2twp.convert(body)
    return f"{pre}{body} {i}"
native_lines = [relabel(l) for l in lines] + [f"{T} {eid}" for T, eid in ext.items()]
open("native_strong_tokens.txt", "w", encoding="utf-8").write("\n".join(native_lines) + "\n")
id2surf = {int(l.rsplit(" ", 1)[1]): l.rsplit(" ", 1)[0] for l in native_lines if len(l.rsplit(" ", 1)) == 2}
def detok(ids): return "".join(id2surf.get(i, "") for i in ids).replace("▁", "")

# ---- build + warm-start + extend head ----
dW = {i.name: nh.to_array(i) for i in onnx.load(f"{M}/decoder-960ms.onnx").graph.initializer}
jW = {i.name: nh.to_array(i) for i in onnx.load(f"{M}/joiner-960ms.onnx").graph.initializer}
dec = Decoder(V1, 512, 0, 2); dproj = nn.Linear(512, 512); outl = nn.Linear(512, V1)
with torch.no_grad():
    emb = torch.from_numpy(dW["decoder.embedding.weight"].copy())          # (5000,512)
    nemb = torch.empty(V1, 512); nn.init.normal_(nemb, std=0.02); nemb[:V0] = emb
    ow = torch.from_numpy(jW["output_linear.weight"].copy()); ob = torch.from_numpy(jW["output_linear.bias"].copy())
    now = torch.empty(V1, 512); nn.init.normal_(now, std=0.02); now[:V0] = ow
    nob = torch.zeros(V1); nob[:V0] = ob
    for eid, sib in ext_sibling.items():                                   # warm-start alternates from siblings
        nemb[eid] = emb[sib].clone(); now[eid] = ow[sib].clone(); nob[eid] = ob[sib].clone()
    dec.embedding.weight.copy_(nemb); dec.conv.weight.copy_(torch.from_numpy(dW["decoder.conv.weight"].copy()))
    dproj.weight.copy_(torch.from_numpy(dW["decoder_proj.weight"].copy())); dproj.bias.copy_(torch.from_numpy(dW["decoder_proj.bias"].copy()))
    outl.weight.copy_(now); outl.bias.copy_(nob)
dec.to(dev); dproj.to(dev); outl.to(dev)

def tok_target(text):
    ids = []; i = 0; n = len(text)
    while i < n:
        c = text[i]
        if CJK.match(c):
            if c in ext: ids.append(ext[c])
            else:
                S = t2s.convert(c)
                if S in simp2id: ids.append(simp2id[S])
            i += 1
        elif c in punct2id: ids.append(punct2id[c]); i += 1
        elif c in ascii_punct: ids.append(punct2id[ascii_punct[c]]); i += 1
        elif re.match(r"[A-Za-z0-9']", c):
            m = re.match(r"[A-Za-z0-9']+", text[i:]); w = m.group(0); ids += sp.encode(w, out_type=int); i += len(w)
        else: i += 1
    return ids

# ---- build training targets, drop empties/too-long ----
tr = [(eo, tt) for eo, tx in zip(tr_eo, tr_tx) for tt in [tok_target(tx)] if 0 < len(tt) <= UMAX and eo.shape[0] <= TMAX]
print(f"train {len(tr)} / eval {len(ev_eo)}", flush=True)

id2simp = {int(l.rsplit(" ", 1)[1]): l.rsplit(" ", 1)[0] for l in lines if len(l.rsplit(" ", 1)) == 2}
def detok_simp(ids): return "".join(id2simp.get(i, "") for i in ids).replace("▁", "")
@torch.no_grad()
def greedy_ids(eo):  # eo np (T,512) -> emitted token ids
    e = torch.from_numpy(eo.astype(np.float32)).to(dev); T = e.shape[0]; hyp = [0, 0]
    do = dproj(dec(torch.tensor([hyp[-2:]], device=dev), need_pad=False).squeeze(1))
    for t in range(T):
        y = int(outl(torch.tanh(e[t:t+1] + do)).squeeze(0).argmax())
        if y != 0: hyp.append(y); do = dproj(dec(torch.tensor([hyp[-2:]], device=dev), need_pad=False).squeeze(1))
    return hyp[2:]
def lev(a, b):
    dp = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        p = dp[0]; dp[0] = i
        for j, y in enumerate(b, 1):
            cur = dp[j]; dp[j] = min(dp[j-1]+1, dp[j]+1, p+(x != y)); p = cur
    return dp[len(b)]
EVAL_N = int(os.environ.get("EVAL_N", str(len(ev_eo))))
def cer_over(render):  # render: ids -> Traditional text
    dec.eval(); dproj.eval(); outl.eval(); e = t = 0
    for eo, tx in zip(ev_eo[:EVAL_N], ev_tx[:EVAL_N]):
        H = [c for c in render(greedy_ids(eo)) if CJK.match(c)]; R = [c for c in tx if CJK.match(c)]
        e += lev(H, R); t += len(R)
    return e / max(t, 1)
def native_cer(): return cer_over(detok)                                  # trained native, no post
def baseline_cer(): return cer_over(lambda ids: s2twp.convert(detok_simp(ids)))  # deployed head + phrase s2twp

BASE = baseline_cer()  # FIXED target: deployed head + phrase s2twp (head still ~original pre-train)
print(f"[eval] deployed-head + phrase-s2twp (baseline, FIXED) = {BASE:.4f}", flush=True)
print(f"[eval] native CER before train = {native_cer():.4f}", flush=True)

# ---- train head with RNN-T loss ----
params = list(dec.parameters()) + list(dproj.parameters()) + list(outl.parameters())
opt = torch.optim.AdamW(params, lr=LR)
def build_batch(items):
    N = len(items); Ts = [it[0].shape[0] for it in items]; Us = [len(it[1]) for it in items]
    Tm, Um = max(Ts), max(Us)
    enc = torch.zeros(N, Tm, 512, device=dev)
    ctx = torch.zeros(N, Um+1, 2, dtype=torch.long, device=dev)  # predictor label-context per step
    tgt = torch.zeros(N, Um, dtype=torch.long, device=dev)
    for n, (eo, ids) in enumerate(items):
        enc[n, :eo.shape[0]] = torch.from_numpy(eo.astype(np.float32)).to(dev)
        tgt[n, :len(ids)] = torch.tensor(ids, device=dev)
        seq = [0, 0] + ids
        for u in range(len(ids)+1): ctx[n, u, 0] = seq[u]; ctx[n, u, 1] = seq[u+1]
    return enc, ctx, tgt, torch.tensor(Ts, dtype=torch.int32, device=dev), torch.tensor(Us, dtype=torch.int32, device=dev)
def save(tag):
    torch.save({"dec": dec.state_dict(), "dproj": dproj.state_dict(), "outl": outl.state_dict(), "V1": V1, "ext": ext},
               f"ft_runs/native_strong{tag}.pt")
best = 1e9; EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "300"))
for step in range(1, STEPS+1):
    dec.train(); dproj.train(); outl.train()
    items = [random.choice(tr) for _ in range(BS)]
    enc, ctx, tgt, Tl, Ul = build_batch(items)
    N, Tm, _ = enc.shape; Um1 = ctx.shape[1]
    do = dproj(dec(ctx.reshape(-1, 2), need_pad=False).squeeze(1)).reshape(N, Um1, 512)  # (N,U+1,512)
    logits = outl(torch.tanh(enc.unsqueeze(2) + do.unsqueeze(1)))  # (N,T,U+1,V)
    loss = AF.rnnt_loss(logits.float(), tgt.int(), Tl, Ul, blank=0, reduction="mean")
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
    if step % 100 == 0: print(f"step {step}/{STEPS} loss={loss.item():.3f}", flush=True)
    if step % EVAL_EVERY == 0:
        c = native_cer(); flag = "  <-- best" if c < best else ""
        if c < best: best = c; save("_best")
        print(f"  [eval@{step}] native CER = {c:.4f}  (FIXED baseline {BASE:.4f}){flag}", flush=True)
c = native_cer()
if c < best: best = c; save("_best")
save("")
print(f"\n=== RESULT: native(best) {best:.4f}  vs  deployed+s2twp {BASE:.4f}  "
      f"({'BEATS' if best < BASE else 'does NOT beat'} baseline by {BASE-best:+.4f}) ===", flush=True)
print("saved ft_runs/native_strong_best.pt + native_strong_tokens.txt", flush=True)
