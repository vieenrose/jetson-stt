"""Cleaner labels via Breeze-ASR-25 (Taiwan SOTA, stronger teacher than X-ASR).
Batched relabel of YouTube+IVOD segment wavs -> Traditional -> t2s Simplified -> breeze_manifest.tsv for kd7.
Filters obvious junk (empty / too-short / repetition)."""
import os, glob, re, time
import numpy as np, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, AutoProcessor
from opencc import OpenCC

HFT=os.environ.get("HFT"); BS=int(os.environ.get("BBS","16")); t2s=OpenCC("t2s"); CJK=re.compile(r"[㐀-鿿]")
proc=AutoProcessor.from_pretrained("MediaTek-Research/Breeze-ASR-25", token=HFT)
model=WhisperForConditionalGeneration.from_pretrained("MediaTek-Research/Breeze-ASR-25", token=HFT, torch_dtype=torch.float16).to("cuda").eval()
print("Breeze loaded", flush=True)
def junk(t):
    z=[c for c in t if CJK.match(c)]
    if len(z)<3: return True
    for c in set(z):  # repetition / hallucination
        if t.count(c)>=8 and t.count(c)>len(t)*0.4: return True
    return False
wavs=sorted(glob.glob("yt_wavs/*.wav")+glob.glob("ivod_wavs/*.wav"))
wavs=[w for w in wavs if "_tmp_" not in w]
print(f"relabeling {len(wavs)} segments with Breeze (batch {BS})", flush=True)
man=open("breeze_manifest.tsv","w",encoding="utf-8"); n=0; kept=0; t0=time.time()
for i in range(0,len(wavs),BS):
    batch=wavs[i:i+BS]; arrs=[]
    for w in batch:
        a,sr=sf.read(w,dtype="float32"); arrs.append(a.mean(1) if a.ndim>1 else a)
    try:
        feats=proc(arrs,sampling_rate=16000,return_tensors="pt").input_features.to("cuda").half()
        with torch.no_grad(): ids=model.generate(feats,language="zh",task="transcribe",max_new_tokens=120)
        txts=proc.batch_decode(ids,skip_special_tokens=True)
    except Exception as e:
        print("batch skip",str(e)[:50],flush=True); continue
    for w,t in zip(batch,txts):
        n+=1; t=t.strip()
        if junk(t): continue
        man.write(f"{os.path.abspath(w)}\t{t2s.convert(t)}\n"); kept+=1
    if (i//BS)%50==0:
        man.flush(); el=time.time()-t0
        print(f"  {n}/{len(wavs)} ({kept} kept) {el:.0f}s",flush=True)
man.close()
print(f"DONE Breeze relabel: {kept}/{n} kept -> breeze_manifest.tsv ({(time.time()-t0)/60:.0f} min)",flush=True)
