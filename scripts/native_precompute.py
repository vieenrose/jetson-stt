"""Precompute frozen encoder_out (960ms strong encoder) + Traditional targets for native head training."""
import os, sys, io, csv, glob, tarfile
sys.path[:0] = ["recipe/X-ASR-zh-en/zipformer", "icefall"]
import numpy as np, soundfile as sf, librosa
from enc_runner import Enc

M = "chunks/960"
enc = Enc(f"{M}/encoder-960ms.onnx", threads=4)
N_CV = int(os.environ.get("N_CV", "1500"))

def feats_texts(items, out):
    """items: list of (id, samples, text). Save encoder_out + text."""
    eos = []; txts = []; ids = []
    for k, (cid, a, t) in enumerate(items):
        try:
            eo = enc.encode_full(a).squeeze(0).numpy().astype(np.float16)  # (T',512) fp16 to save space
        except Exception as e:
            print("skip", cid, str(e)[:50], flush=True); continue
        if eo.shape[0] < 1: continue
        eos.append(eo); txts.append(t); ids.append(cid)
        if (k+1) % 200 == 0: print(f"  {out}: {k+1}/{len(items)}", flush=True)
    np.savez(out, eo=np.array(eos, dtype=object), txt=np.array(txts, dtype=object), ids=np.array(ids, dtype=object))
    print(f"saved {out}: {len(eos)} clips", flush=True)

# 1) NTUML tw_train (Traditional, space-separated)
ntu = {}
for l in open("ftdata/tw_train.tsv", encoding="utf-8"):
    if "\t" in l: u, t = l.rstrip("\n").split("\t", 1); ntu[u] = t
items = []
for w in sorted(glob.glob("ftdata/tw_train/*.wav")):
    u = os.path.basename(w)[:-4]
    if u in ntu:
        a, sr = sf.read(w, dtype="float32"); a = a.mean(1) if a.ndim > 1 else a
        items.append((u, a, ntu[u]))
print(f"NTUML train: {len(items)}", flush=True)
feats_texts(items, "feat_ntuml_train.npz")

# 2) CV zh-TW train (Traditional)
TAR = open("cv_train_tar_path.txt").read().strip(); TSV = open("cv_train_tsv_path.txt").read().strip()
sent = {}
with open(TSV, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"): sent[os.path.basename(row["path"])] = row["sentence"].strip()
items = []; n = 0
with tarfile.open(TAR) as tar:
    for m in tar:
        if n >= N_CV: break
        if not m.name.endswith(".mp3"): continue
        t = sent.get(os.path.basename(m.name))
        if not t: continue
        try: a, _ = librosa.load(io.BytesIO(tar.extractfile(m).read()), sr=16000, mono=True)
        except Exception: continue
        if len(a) < 1600: continue
        items.append((os.path.basename(m.name), a, t)); n += 1
print(f"CV train: {len(items)}", flush=True)
feats_texts(items, "feat_cv_train.npz")

# 3) Eval = the 500 CV test clips (already cached audio)
z = np.load("cv_audio_cache_500.npz", allow_pickle=True); A = list(z["a"]); R = list(z["r"])
items = [(f"ev{i}", A[i], R[i]) for i in range(len(A))]
print(f"CV eval: {len(items)}", flush=True)
feats_texts(items, "feat_cv_eval.npz")
print("DONE precompute", flush=True)
