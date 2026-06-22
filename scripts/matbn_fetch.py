"""Patient download + process of formospeech/matbn (Taiwan Mandarin, multi-source).
Robust file download (not streaming, which hangs here) -> fbank + Simplified text -> matbn_feats shards for kd6."""
import os, io, re
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import numpy as np, soundfile as sf
from opencc import OpenCC
from lhotse import Fbank, FbankConfig

HFT=os.environ.get("HFT"); t2s=OpenCC("t2s"); CJK=re.compile(r"[㐀-鿿]")
FB=Fbank(FbankConfig(sampling_rate=16000,num_mel_bins=80,snip_edges=False,dither=0.0))
def feat(a): return np.asarray(FB.extract(np.ascontiguousarray(a.astype(np.float32)),16000)).astype(np.float16)
os.makedirs("matbn_feats",exist_ok=True)
def decode_audio(au):
    # HF Audio in parquet: dict {'bytes':..., 'path':...} or {'array','sampling_rate'}
    if isinstance(au,dict):
        if au.get("array") is not None:
            a=np.asarray(au["array"],dtype=np.float32); sr=au.get("sampling_rate",16000)
        elif au.get("bytes"):
            a,sr=sf.read(io.BytesIO(au["bytes"]),dtype="float32")
        else: return None
    else:
        a,sr=sf.read(io.BytesIO(au),dtype="float32")
    if a.ndim>1: a=a.mean(1)
    if sr!=16000:
        import librosa; a=librosa.resample(a,orig_sr=sr,target_sr=16000)
    return a
buf_f,buf_t=[],[]; shard=0; hours=0.0; n=0
def flush():
    global shard,buf_f,buf_t
    if not buf_f: return
    np.savez(f"matbn_feats/shard_{shard:03d}.npz",eo=np.array(buf_f,dtype=object),txt=np.array(buf_t,dtype=object))
    print(f"  matbn shard_{shard:03d}: {len(buf_f)} clips, {hours:.1f}h",flush=True); shard+=1; buf_f,buf_t=[],[]
for i in range(29):
    try:
        f=hf_hub_download("formospeech/matbn",f"cmn/train-{i:05d}-of-00029.parquet",repo_type="dataset",token=HFT)
    except Exception as e:
        print(f"shard {i} download fail: {str(e)[:60]}",flush=True); continue
    df=pq.read_table(f).to_pandas()
    for _,row in df.iterrows():
        try: a=decode_audio(row["audio"])
        except Exception: a=None
        if a is None or len(a)<1600: continue
        txt=t2s.convert(str(row.get("text") or "").strip())
        if not any(CJK.match(c) or c.isalnum() for c in txt): continue
        ff=feat(a)
        if not (5<=ff.shape[0]<=1500): continue
        buf_f.append(ff); buf_t.append(txt); hours+=len(a)/16000/3600; n+=1
        if len(buf_f)>=2000: flush()
    print(f"shard {i} done; cum {n} clips {hours:.1f}h",flush=True)
flush()
print(f"DONE matbn: {n} clips, {hours:.1f}h Taiwan Mandarin -> matbn_feats/",flush=True)
