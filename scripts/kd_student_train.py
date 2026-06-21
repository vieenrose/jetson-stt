"""Half-size distillation: train an 82M student on teacher pseudo-labels (k2 pruned RNN-T), slice-init from base."""
import os, sys, glob, random, re, argparse
sys.path[:0] = ["recipe/X-ASR-zh-en/zipformer", "icefall"]
import numpy as np, torch, torch.nn as nn, k2, soundfile as sf
import train as T
import sentencepiece as spm
from opencc import OpenCC
from lhotse import Fbank, FbankConfig
from huggingface_hub import hf_hub_download
from beam_search import greedy_search_batch

STEPS = int(os.environ.get("STEPS", "8000")); BS = int(os.environ.get("BS", "12"))
LR = float(os.environ.get("LR", "4e-4")); WARM = int(os.environ.get("WARM", "500"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "1000")); dev = "cuda"
random.seed(0); torch.manual_seed(0)
t2s = OpenCC("t2s"); CJK = re.compile(r"[㐀-鿿豈-﫿]")

STU = dict(num_encoder_layers="2,2,4,5,4,2", encoder_dim="160,192,384,512,384,192",
    feedforward_dim="384,512,1024,1280,1024,512", num_heads="4,4,4,8,4,4",
    encoder_unmasked_dim="160,160,192,256,192,160", query_head_dim="32", value_head_dim="12",
    pos_head_dim="4", pos_dim="48", cnn_module_kernel="31,31,15,15,15,31",
    downsampling_factor="1,2,4,8,4,2", causal="1", chunk_size="16,32,64,-1", left_context_frames="64,128,256,-1")
def build(arch):
    p = argparse.ArgumentParser(); T.add_model_arguments(p); argv = []
    for k, v in arch.items(): argv += [f"--{k.replace('_','-')}", v]
    a = p.parse_args(argv); pr = T.get_params(); pr.update(vars(a))
    pr.blank_id = 0; pr.vocab_size = 5000; pr.context_size = 2; pr.decoder_dim = 512; pr.joiner_dim = 512
    pr.use_transducer = True; pr.use_ctc = False
    return T.get_model(pr)

m = build(STU)
print(f"student params: {sum(p.numel() for p in m.parameters())/1e6:.1f}M", flush=True)
# slice-init from base pretrained.pt (leading sub-block of each matching tensor)
bp = hf_hub_download("GilgameshWind/X-ASR-zh-en", "streaming_exp/pretrained.pt")
bsd = torch.load(bp, map_location="cpu", weights_only=False)
base = bsd.get("model_avg", bsd.get("model", bsd))
ssd = m.state_dict(); copied = 0
with torch.no_grad():
    for n, p in ssd.items():
        b = base.get(n)
        if b is not None and p.dim() == b.dim() and all(s <= bs for s, bs in zip(p.shape, b.shape)):
            p.copy_(b[tuple(slice(0, s) for s in p.shape)]); copied += 1
print(f"slice-init: copied {copied}/{len(ssd)} tensors from base", flush=True)
m.to(dev)

sp = spm.SentencePieceProcessor(); sp.load("recipe/X-ASR-zh-en/zipformer/data/lang_5000_with_punctuation/bpe_punc.model")
def space_cjk(t): return re.sub(r"\s+", " ", "".join((" "+c+" ") if CJK.match(c) else c for c in t)).strip()
def tok(text): return sp.encode(space_cjk(t2s.convert(text)), out_type=int)

_FB = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80, high_freq=-400, snip_edges=False, dither=0.0))
def fbank_paths(paths):
    feats = []
    for w in paths:
        a, sr = sf.read(w, dtype="float32"); a = a.mean(1) if a.ndim > 1 else a
        feats.append(torch.from_numpy(np.asarray(_FB.extract(np.ascontiguousarray(a), 16000))).float())
    L = torch.tensor([f.shape[0] for f in feats]); Tm = int(L.max())
    x = torch.zeros(len(feats), Tm, 80)
    for i, f in enumerate(feats): x[i, :f.shape[0]] = f
    return x.to(dev), L.to(dev)
def fbank_arr(a):
    f = torch.from_numpy(np.asarray(_FB.extract(np.ascontiguousarray(a.astype(np.float32)), 16000))).float()
    return f.unsqueeze(0).to(dev), torch.tensor([f.shape[0]]).to(dev)

# data
train = []
for l in open("kd_manifest.tsv", encoding="utf-8"):
    if "\t" in l:
        w, t = l.rstrip("\n").split("\t", 1); ids = tok(t)
        if 0 < len(ids) <= 80: train.append((w, ids))
print(f"train pairs: {len(train)}", flush=True)
ev = np.load("cv_audio_cache_500.npz", allow_pickle=True); EA = list(ev["a"]); ER = list(ev["r"])

def lev(a, b):
    dp = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        p = dp[0]; dp[0] = i
        for j, y in enumerate(b, 1):
            cur = dp[j]; dp[j] = min(dp[j-1]+1, dp[j]+1, p+(x != y)); p = cur
    return dp[len(b)]
sym = {}  # id->piece for detok
for i in range(sp.get_piece_size()): sym[i] = sp.id_to_piece(i)
@torch.no_grad()
def cer():
    m.eval(); e = t = 0
    for i in range(0, len(EA), 16):
        batch = EA[i:i+16]
        xs = [fbank_arr(a) for a in batch]
        Tm = max(x.shape[1] for x, _ in xs); x = torch.zeros(len(batch), Tm, 80, device=dev); L = torch.zeros(len(batch), dtype=torch.long, device=dev)
        for j, (xi, li) in enumerate(xs): x[j, :xi.shape[1]] = xi[0]; L[j] = li
        eo, el = m.forward_encoder(x, L)
        for j, h in enumerate(greedy_search_batch(model=m, encoder_out=eo, encoder_out_lens=el)):
            txt = "".join(sym.get(i, "") for i in h).replace("▁", "")
            H = [c for c in t2s.convert(txt) if CJK.match(c)]; R = [c for c in t2s.convert(ER[i+j]) if CJK.match(c)]
            e += lev(H, R); t += len(R)
    m.train(); return e / max(t, 1)

opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-2)
def set_lr(step):
    lr = LR * min(1.0, step / WARM) * (max(0.1, (STEPS - step) / STEPS) if step > WARM else 1.0)
    for g in opt.param_groups: g["lr"] = lr
print(f"[eval@0] student CER = {cer():.4f}  (teacher ~0.064)", flush=True)
m.train(); best = 1e9
for step in range(1, STEPS+1):
    set_lr(step)
    items = [random.choice(train) for _ in range(BS)]
    x, xl = fbank_paths([w for w, _ in items])
    y = k2.RaggedTensor([ids for _, ids in items]).to(dev)
    try:
        simple, pruned, *_ = m(x, xl, y, prune_range=5, am_scale=0.0, lm_scale=0.0)
        loss = 0.5 * simple + pruned
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    except Exception as ex:
        print("step", step, "skip", str(ex)[:80], flush=True); continue
    if step % 100 == 0:
        nt = sum(len(i) for _, i in items)
        print(f"step {step}/{STEPS} lr={opt.param_groups[0]['lr']:.1e} loss/tok={pruned.item()/max(nt,1):.3f}", flush=True)
    if step % EVAL_EVERY == 0:
        c = cer(); tag = "  <-- best" if c < best else ""
        if c < best: best = c; torch.save({"model": m.state_dict(), "arch": STU}, "ft_runs/kd_student_best.pt")
        print(f"  [eval@{step}] student CER = {c:.4f}  (teacher ~0.064){tag}", flush=True)
print(f"\n=== KD POC: best student CER {best:.4f} vs teacher 0.064 (82M = 52% of teacher) ===", flush=True)
