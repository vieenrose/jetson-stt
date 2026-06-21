#!/usr/bin/env python3
"""Sweep deployed model chunk sizes (480/960/1920ms) + s2twp on the same 500 CV17 zh-TW clips.
Reports Traditional-output CER (the attendant metric) and compute RTF (GB10 CPU, 2 threads)."""
import os, re, io, csv, json, random, tarfile, time
import numpy as np, librosa, sherpa_onnx
from opencc import OpenCC

N = int(os.environ.get("N", "500"))
s2twp = OpenCC("s2twp"); CJK = re.compile(r"[㐀-鿿]"); WORD = re.compile(r"[A-Za-z0-9']+")
random.seed(0)
TAR = open("cv_test_tar_path.txt").read().strip(); TSV = open("cv_test_tsv_path.txt").read().strip()

def load(d, ms):
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/tokens.txt", encoder=f"{d}/encoder-{ms}ms.onnx",
        decoder=f"{d}/decoder-{ms}ms.onnx", joiner=f"{d}/joiner-{ms}ms.onnx",
        num_threads=2, provider="cpu", decoding_method="greedy_search",
        sample_rate=16000, feature_dim=80)

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
refs = json.load(open(f"cv_eval_cache_{N}.json"))["refs"]

# load audio once (re-read tar), keep arrays for all chunk decoders
AUD = "cv_audio_cache_%d.npz" % N
if os.path.exists(AUD):
    z = np.load(AUD, allow_pickle=True); audio = list(z["a"]); rchk = list(z["r"])
else:
    audio = []; rchk = []; n = 0
    with tarfile.open(TAR) as tar:
        for m in tar:
            if n >= N: break
            if not m.name.endswith(".mp3"): continue
            ref = sent.get(os.path.basename(m.name))
            if not ref: continue
            try: a, _ = librosa.load(io.BytesIO(tar.extractfile(m).read()), sr=16000, mono=True)
            except Exception: continue
            if len(a) < 1600: continue
            audio.append(a.astype("float32")); rchk.append(ref); n += 1
    np.savez(AUD, a=np.array(audio, dtype=object), r=np.array(rchk, dtype=object))
assert rchk == refs, "alignment mismatch"
dur = sum(len(a) for a in audio) / 16000.0
print(f"{len(audio)} clips, {dur/60:.1f} min audio\n", flush=True)

def cer(hyp, ref):
    per = []; e = t = 0
    for h, r in zip(hyp, ref):
        H = [c for c in toks(s2twp.convert(h)) if CJK.match(c)]
        R = [c for c in toks(r) if CJK.match(c)]
        d = lev(H, R); e += d; t += len(R); per.append((d, max(len(R), 1)))
    vals = []
    for _ in range(1000):
        s = [per[random.randrange(len(per))] for _ in per]
        vals.append(sum(x[0] for x in s) / max(sum(x[1] for x in s), 1))
    vals.sort(); return e/t, (vals[25], vals[975])

print(f"{'chunk':8s} {'latency':8s} {'Trad CER (+s2twp)':>22s} {'compute RTF@2thr':>16s}")
for ms in ["480", "960", "1920"]:
    d = f"chunks/{ms}"
    if not os.path.exists(f"{d}/encoder-{ms}ms.onnx"): print(f"{ms}: missing"); continue
    r = load(d, ms)
    t0 = time.time(); hyp = [dec(r, a) for a in audio]; rtf = (time.time() - t0) / dur
    c, ci = cer(hyp, refs)
    print(f"{ms+'ms':8s} {ms+'ms':8s} {c:.3f} [{ci[0]:.3f},{ci[1]:.3f}]      {rtf:>10.3f}", flush=True)
