"""kd6: Taiwan-at-scale corpus for the small-bilingual student.
Sources: YouTube (yt_manifest) + IVOD (ivod_manifest) + CV-zh-TW train + NTUML -> (fbank, Simplified text) shards.
All Taiwan-accent, recognition-only use. fbank = small-bilingual default high_freq."""
import os, sys, io, csv, glob, tarfile, re
sys.path[:0] = ["icefall"]
import numpy as np, soundfile as sf, librosa
from opencc import OpenCC
from lhotse import Fbank, FbankConfig

t2s=OpenCC("t2s"); CJK=re.compile(r"[㐀-鿿]")
FB=Fbank(FbankConfig(sampling_rate=16000,num_mel_bins=80,snip_edges=False,dither=0.0))
def feat(a): return np.asarray(FB.extract(np.ascontiguousarray(a.astype(np.float32)),16000)).astype(np.float16)
os.makedirs("kd7_feats",exist_ok=True)
buf_f,buf_t=[],[]; shard=0; hours=0.0; n=0; bysrc={}
def flush():
    global shard,buf_f,buf_t
    if not buf_f: return
    np.savez(f"kd7_feats/shard_{shard:03d}.npz",eo=np.array(buf_f,dtype=object),txt=np.array(buf_t,dtype=object))
    print(f"  shard_{shard:03d}: {len(buf_f)} clips, {hours:.1f}h",flush=True); shard+=1; buf_f,buf_t=[],[]
def add(a,text,src):
    global hours,n
    if a is None or len(a)<1600: return
    text=t2s.convert((text or "").strip())
    if not any(CJK.match(c) or c.isalnum() for c in text): return
    f=feat(a)
    if not (5<=f.shape[0]<=1500): return
    buf_f.append(f); buf_t.append(text); hours+=len(a)/16000/3600; n+=1; bysrc[src]=bysrc.get(src,0)+1
    if len(buf_f)>=3000: flush()

# manifest sources (wav, text): YouTube + IVOD
for mani,src in [("breeze_manifest.tsv","breeze")]:
    if not os.path.exists(mani): continue
    seen=set()
    for l in open(mani,encoding="utf-8"):
        if "\t" not in l: continue
        w,t=l.rstrip("\n").split("\t",1)
        if w in seen or not os.path.exists(w): continue
        seen.add(w)
        try: a,_=sf.read(w,dtype="float32"); add(a.mean(1) if a.ndim>1 else a, t, src)
        except Exception: pass
    print(f"after {src}: {n} ({hours:.1f}h)",flush=True)
# CV-zh-TW train (GT)
if os.path.exists("cv_train_tar_path.txt"):
    TAR=open("cv_train_tar_path.txt").read().strip(); TSV=open("cv_train_tsv_path.txt").read().strip()
    sent={}
    for r in csv.DictReader(open(TSV,encoding="utf-8"),delimiter="\t"): sent[os.path.basename(r["path"])]=r["sentence"].strip()
    with tarfile.open(TAR) as tar:
        for m in tar:
            if not m.name.endswith(".mp3"): continue
            t=sent.get(os.path.basename(m.name))
            if not t: continue
            try: a,_=librosa.load(io.BytesIO(tar.extractfile(m).read()),sr=16000,mono=True); add(a,t,"cv")
            except Exception: pass
    print(f"after CV: {n} ({hours:.1f}h)",flush=True)
# NTUML (GT)
ntu={l.split('\t')[0]:l.split('\t',1)[1].strip() for l in open("ftdata/tw_train.tsv") if "\t" in l}
for w in sorted(glob.glob("ftdata/tw_train/*.wav")):
    u=os.path.basename(w)[:-4]
    if u in ntu: a,sr=sf.read(w,dtype="float32"); add(a.mean(1) if a.ndim>1 else a, ntu[u], "ntuml")
flush()
print(f"DONE kd7: {n} clips, {hours:.1f}h Taiwan -> {shard} shards | by source: {bysrc}",flush=True)
