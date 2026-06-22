"""kd5: assemble a TAIWAN-ONLY corpus (no Mainland AISHELL) for the small-bilingual student.
CV-zh-TW + formosaspeech + IVOD(gathered) + NTUML + ASCEND -> (fbank fp16, Simplified text) shards.
Tests whether Taiwan-accent data (vs kd4's Mainland-diluted 167h) finally improves the CV-zh-TW eval."""
import os, sys, io, csv, glob, tarfile, re
sys.path[:0] = ["icefall"]
import numpy as np, soundfile as sf, librosa
from opencc import OpenCC
from datasets import load_dataset, Audio
from lhotse import Fbank, FbankConfig

HFT = os.environ.get("HFT")
t2s = OpenCC("t2s"); CJK = re.compile(r"[㐀-鿿]")
FB = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80, snip_edges=False, dither=0.0))  # small-bilingual default high_freq
def feat(a): return np.asarray(FB.extract(np.ascontiguousarray(a.astype(np.float32)), 16000)).astype(np.float16)
os.makedirs("kd5_feats", exist_ok=True)
buf_f, buf_t = [], []; shard = 0; hours = 0.0; n = 0; bysrc = {}
def flush():
    global shard, buf_f, buf_t
    if not buf_f: return
    np.savez(f"kd5_feats/shard_{shard:03d}.npz", eo=np.array(buf_f, dtype=object), txt=np.array(buf_t, dtype=object))
    print(f"  shard_{shard:03d}: {len(buf_f)} clips, {hours:.1f}h cum", flush=True); shard += 1; buf_f, buf_t = [], []
def add(a, text, src):
    global hours, n
    if a is None or len(a) < 1600: return
    text = t2s.convert((text or "").strip())
    if not any(CJK.match(c) or c.isalnum() for c in text): return
    f = feat(a)
    if not (5 <= f.shape[0] <= 1500): return
    buf_f.append(f); buf_t.append(text); hours += len(a)/16000/3600; n += 1; bysrc[src] = bysrc.get(src,0)+1
    if len(buf_f) >= 2000: flush()

# 1) IVOD (gathered Taiwan parliamentary; pseudo-labeled by X-ASR)
iv = {}
if os.path.exists("ivod_manifest.tsv"):
    for l in open("ivod_manifest.tsv", encoding="utf-8"):
        if "\t" in l: w,t=l.rstrip("\n").split("\t",1); iv[w]=t
for w in glob.glob("ivod_wavs/*.wav"):
    if "_tmp_" in w: continue
    t = iv.get(os.path.abspath(w))
    if t:
        a,sr = sf.read(w,dtype="float32"); add(a.mean(1) if a.ndim>1 else a, t, "ivod")
print(f"after IVOD: {n} ({hours:.1f}h)", flush=True)
# 2) CV-zh-TW (in-domain Taiwan, GT)
TAR=open("cv_train_tar_path.txt").read().strip(); TSV=open("cv_train_tsv_path.txt").read().strip()
sent={}
for r in csv.DictReader(open(TSV,encoding="utf-8"),delimiter="\t"): sent[os.path.basename(r["path"])]=r["sentence"].strip()
with tarfile.open(TAR) as tar:
    for m in tar:
        if not m.name.endswith(".mp3"): continue
        t=sent.get(os.path.basename(m.name))
        if not t: continue
        try: a,_=librosa.load(io.BytesIO(tar.extractfile(m).read()),sr=16000,mono=True)
        except Exception: continue
        add(a,t,"cv")
print(f"after CV-zh-TW: {n} ({hours:.1f}h)", flush=True)
# 3) NTUML (Taiwan CS, GT)
ntu={l.split('\t')[0]:l.split('\t',1)[1].strip() for l in open("ftdata/tw_train.tsv") if "\t" in l}
for w in sorted(glob.glob("ftdata/tw_train/*.wav")):
    u=os.path.basename(w)[:-4]
    if u in ntu: a,sr=sf.read(w,dtype="float32"); add(a.mean(1) if a.ndim>1 else a, ntu[u], "ntuml")
print(f"after NTUML: {n} ({hours:.1f}h)", flush=True)
# 4) formosaspeech (Taiwan Mandarin, GT)
try:
    for ex in load_dataset("MediaTek-Research/formosaspeech",split="train",streaming=True,token=HFT).cast_column("audio",Audio(sampling_rate=16000)):
        add(ex["audio"]["array"], ex.get("text") or "", "formosa")
except Exception as e: print("formosa err",str(e)[:60],flush=True)
print(f"after formosaspeech: {n} ({hours:.1f}h)", flush=True)
# 5) ASCEND (zh-en CS, HK accent, GT)
try:
    for ex in load_dataset("CAiRE/ASCEND",split="train",streaming=True,token=HFT).cast_column("audio",Audio(sampling_rate=16000)):
        add(ex["audio"]["array"], ex.get("transcription") or ex.get("sentence") or "", "ascend")
except Exception as e: print("ascend err",str(e)[:60],flush=True)
flush()
print(f"DONE: {n} clips, {hours:.1f}h Taiwan-only -> {shard} shards | by source: {bysrc}", flush=True)
