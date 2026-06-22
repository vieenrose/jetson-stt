"""Gather real Taiwan-Mandarin audio from Legislative Yuan IVOD (g0v API, gov/Art.9 clean license).
Parallel (thread pool): enumerate clips -> curl HLS -> 12s windows -> X-ASR pseudo-label -> manifest.
Resumable: skips IVOD_IDs already downloaded. curl for network (ffmpeg's resolver fails on hinet)."""
import os, json, subprocess, urllib.request, re, glob, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, soundfile as sf, sherpa_onnx
from opencc import OpenCC

TARGET_H = float(os.environ.get("TARGET_H", "400")); WORKERS = int(os.environ.get("WORKERS", "8")); WIN = 12.0
DEP = "/tmp/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"
t2s = OpenCC("t2s"); CJK = re.compile(r"[㐀-鿿]")
os.makedirs("ivod_wavs", exist_ok=True)
_tl = threading.local()
def get_rec():
    if not hasattr(_tl, "rec"):
        _tl.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{DEP}/tokens.txt", encoder=f"{DEP}/encoder.int8.onnx", decoder=f"{DEP}/decoder.onnx",
            joiner=f"{DEP}/joiner.int8.onnx", num_threads=1, provider="cpu", decoding_method="greedy_search",
            sample_rate=16000, feature_dim=80)
    return _tl.rec
def asr(a):
    r = get_rec(); s = r.create_stream(); s.accept_waveform(16000, a.astype("float32"))
    s.accept_waveform(16000, np.zeros(32000, "float32")); s.input_finished()
    while r.is_ready(s): r.decode_stream(s)
    x = r.get_result(s); return (x if isinstance(x, str) else x.text).strip()
def curl_t(u): return subprocess.run(["curl","-s","--max-time","60",u],capture_output=True,text=True,timeout=70).stdout
def curl_b(u): return subprocess.run(["curl","-s","--max-time","120",u],capture_output=True,timeout=130).stdout
def fetch_clip(url, out):
    base = url.rsplit("/",1)[0]; master = curl_t(url)
    var = next((l.strip() for l in master.splitlines() if l.strip().endswith(".m3u8")), None)
    if not var: raise RuntimeError("no variant")
    vbase = f"{base}/{var}".rsplit("/",1)[0]
    segs = [l.strip() for l in curl_t(f"{base}/{var}").splitlines() if l.strip() and not l.startswith("#")]
    if not segs: raise RuntimeError("no segs")
    ts = out+".ts"  # sequential .ts fetch (reliable; clip-level parallelism via worker pool)
    with open(ts,"wb") as o:
        for s in segs:
            b = curl_b(s if s.startswith("http") else f"{vbase}/{s}")
            if b: o.write(b)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",ts,"-ar","16000","-ac","1",out],timeout=300,check=True)
    os.remove(ts)

def enum_clips(n):
    out=[]; page=1
    while len(out)<n and page<=400:
        try:
            req=urllib.request.Request(f"https://ly.govapi.tw/v2/ivods?limit=100&page={page}",headers={"User-Agent":"research"})
            d=json.load(urllib.request.urlopen(req,timeout=30))
        except Exception as e: print("api err",str(e)[:50],flush=True); break
        recs=d.get("ivods",d.get("data",[]))
        if not recs: break
        for it in recs:
            dur=it.get("影片長度",0); u=it.get("video_url","")
            if it.get("影片種類")=="Clip" and 40<=dur<=900 and u.endswith(".m3u8"): out.append((str(it.get("IVOD_ID")),u,dur))
        page+=1
    return out

def work(clip):
    cid,url,dur=clip
    if state["h"]>=TARGET_H: return
    raw=f"ivod_wavs/_tmp_{cid}.wav"
    try:
        fetch_clip(url,raw); a,_=sf.read(raw,dtype="float32")
    except Exception:
        if os.path.exists(raw): os.remove(raw)
        return
    W=int(WIN*16000); rows=[]
    for k in range(0,len(a),W):
        seg=a[k:k+W]
        if len(seg)<16000 or float(np.abs(seg).max())<0.01: continue
        txt=asr(seg); zh=[c for c in txt if CJK.match(c)]
        if len(zh)<3: continue
        sp=f"ivod_wavs/{cid}_{k//W}.wav"; sf.write(sp,seg,16000); rows.append((sp,t2s.convert(txt),len(seg)/16000))
    os.remove(raw)
    with lock:
        for sp,txt,dur_s in rows:
            man.write(f"{os.path.abspath(sp)}\t{txt}\n"); state["h"]+=dur_s/3600; state["n"]+=1
        man.flush()
        if state["n"]%200<len(rows): print(f"  {state['n']} segs, {state['h']:.1f}h",flush=True)
if __name__ == "__main__":
    done={os.path.basename(w).split("_")[0] for w in glob.glob("ivod_wavs/*.wav")}
    clips=[c for c in enum_clips(int(TARGET_H*3600/200)+len(done)+500) if c[0] not in done]
    print(f"enumerated {len(clips)} new clips ({len(done)} already done)", flush=True)
    lock=threading.Lock(); man=open("ivod_manifest.tsv","a",encoding="utf-8")
    state={"h":sum(sf.info(w).duration for w in glob.glob("ivod_wavs/*.wav"))/3600,"n":len(glob.glob('ivod_wavs/*.wav'))}
    print(f"resuming from {state['h']:.1f}h", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs=[ex.submit(work,c) for c in clips]
        for _ in as_completed(futs):
            if state["h"]>=TARGET_H: break
    print(f"DONE: {state['n']} segments, {state['h']:.1f}h Taiwan-Mandarin -> ivod_manifest.tsv", flush=True)
