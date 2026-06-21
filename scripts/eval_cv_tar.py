#!/usr/bin/env python3
"""Benchmark base/FT/native on CV17 zh-TW test (read mp3 from tar)."""
import os, re, io, csv, random, tarfile
import numpy as np, librosa, sherpa_onnx
from opencc import OpenCC

N = int(os.environ.get("N", "500"))
t2s = OpenCC("t2s"); CJK = re.compile(r"[㐀-鿿]"); WORD = re.compile(r"[A-Za-z0-9']+")
random.seed(0)
TAR = open("cv_test_tar_path.txt").read().strip(); TSV = open("cv_test_tsv_path.txt").read().strip()

def load(tok, d):
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tok, encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.onnx",
        joiner=f"{d}/joiner.int8.onnx", num_threads=2, provider="cpu",
        decoding_method="greedy_search", sample_rate=16000, feature_dim=80)

def dec(r, a):
    s = r.create_stream(); s.accept_waveform(16000, a.astype("float32"))
    s.accept_waveform(16000, np.zeros(32000, "float32")); s.input_finished()
    while r.is_ready(s): r.decode_stream(s)
    x = r.get_result(s); return x if isinstance(x, str) else x.text

def toks(s):
    out = []; i = 0
    while i < len(s):
        c = s[i]
        if CJK.match(c): out.append(c); i += 1
        elif c == "'" or ("a" <= c.lower() <= "z") or ("0" <= c <= "9"):
            m = WORD.match(s, i)
            if m: out.append(m.group(0).lower()); i = m.end()
            else: i += 1
        else: i += 1
    return out

def lev(a, b):
    dp = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        p = dp[0]; dp[0] = i
        for j, y in enumerate(b, 1):
            cur = dp[j]; dp[j] = min(dp[j-1]+1, dp[j]+1, p+(x != y)); p = cur
    return dp[len(b)]

sent = {}
with open(TSV, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        sent[os.path.basename(row["path"])] = row["sentence"].strip()
print(f"transcript rows: {len(sent)}", flush=True)

MODELS = {"base": ("pkg/base/tokens.txt", "pkg/base"),
          "FT": ("pkg/ft/tokens.txt", "pkg/ft"),
          "native": ("pkg/ft/tok_s2twp.txt", "pkg/ft")}
recs = {k: load(t, d) for k, (t, d) in MODELS.items()}

import json
CACHE = f"cv_eval_cache_{N}.json"
if os.path.exists(CACHE):
    c = json.load(open(CACHE)); hyps = c["hyps"]; refs = c["refs"]; n = len(refs)
    print(f"loaded cache: {n} clips", flush=True)
else:
  hyps = {k: [] for k in MODELS}; refs = []; n = 0
  with tarfile.open(TAR) as tar:
    for m in tar:
        if n >= N: break
        if not m.name.endswith(".mp3"): continue
        ref = sent.get(os.path.basename(m.name))
        if not ref: continue
        try:
            a, _ = librosa.load(io.BytesIO(tar.extractfile(m).read()), sr=16000, mono=True)
        except Exception:
            continue
        if len(a) < 1600: continue
        refs.append(ref)
        for k in MODELS: hyps[k].append(dec(recs[k], a))
        n += 1
        if n % 100 == 0: print(f"  {n}/{N}", flush=True)
  json.dump({"hyps": hyps, "refs": refs}, open(CACHE, "w"), ensure_ascii=False)
print(f"decoded {n} clips\n", flush=True)

def score(hyp, ref, norm):
    me = mt = ze = zt = 0; per = []
    for h, r in zip(hyp, ref):
        if norm == "simp": h = t2s.convert(h); r = t2s.convert(r)
        ht = toks(h); rt = toks(r); e = lev(ht, rt); t = max(len(rt), 1)
        me += e; mt += t
        hz = [c for c in ht if CJK.match(c)]; rz = [c for c in rt if CJK.match(c)]
        ze += lev(hz, rz); zt += len(rz); per.append((e, t))
    vals = []
    for _ in range(1000):
        s = [per[random.randrange(len(per))] for _ in per]
        vals.append(sum(x[0] for x in s) / max(sum(x[1] for x in s), 1))
    vals.sort()
    return me/mt, (vals[25], vals[975]), ze/max(zt, 1)

print(f"{'model':10s} {'output':12s} {'MER(norm)':>22s} {'zhCER(norm)':>11s} {'raw-CER(Trad ref)':>18s}")
for k in MODELS:
    mer, ci, zc = score(hyps[k], refs, "simp")
    _, _, raw = score(hyps[k], refs, "asis")
    out = "Traditional" if k == "native" else "Simplified"
    print(f"{k:10s} {out:12s} {mer:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  {zc:>9.3f}  {raw:>16.3f}")
