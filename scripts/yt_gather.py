"""Gather Taiwan-Mandarin audio from YouTube (Google CDN = reachable+fast on this HiNet box).
Per video: yt-dlp(deno, DASH itag-140, own resolver -> bypasses sandbox ffmpeg-DNS) -> 16k wav ->
12s windows -> X-ASR pseudo-label -> manifest. Resumable. Bias: TW gov/edu/public channels.
NOTE: research/internal use — for a shipping product mind TW copyright (prefer official gov channels)."""
import os, re, glob, json, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, soundfile as sf, sherpa_onnx
from opencc import OpenCC

TARGET_H=float(os.environ.get("TARGET_H","80")); WORKERS=int(os.environ.get("WORKERS","3")); WIN=12.0
YTDLP="/home/luigi/jetson-stt/.venv/bin/yt-dlp"; ENV=dict(os.environ, PATH="/home/luigi/.local/bin:"+os.environ.get("PATH",""))
DEP="/tmp/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"
t2s=OpenCC("t2s"); CJK=re.compile(r"[㐀-鿿]"); os.makedirs("yt_wavs",exist_ok=True)
SOURCES=[  # broad TW Mandarin coverage (news/talk/lecture/gov/general) — recognition-only use
 "ytsearch80:台灣 新聞 完整版","ytsearch80:台灣 政論 節目 完整","ytsearch60:三立新聞 完整",
 "ytsearch60:國會頻道 立法院 質詢 完整","ytsearch60:台灣 大學 公開課 演講",
 "ytsearch60:台灣 podcast 訪談 中文","ytsearch60:台灣 記者會 完整","ytsearch50:憲法法庭 言詞辯論",
 "ytsearch60:台灣 youtuber 中文 vlog","ytsearch50:台灣 講座 演講 全程",
]
_tl=threading.local()
def rec():
    if not hasattr(_tl,"r"):
        _tl.r=sherpa_onnx.OnlineRecognizer.from_transducer(tokens=f"{DEP}/tokens.txt",encoder=f"{DEP}/encoder.int8.onnx",
            decoder=f"{DEP}/decoder.onnx",joiner=f"{DEP}/joiner.int8.onnx",num_threads=1,provider="cpu",
            decoding_method="greedy_search",sample_rate=16000,feature_dim=80)
    return _tl.r
def asr(a):
    r=rec(); s=r.create_stream(); s.accept_waveform(16000,a.astype("float32")); s.accept_waveform(16000,np.zeros(32000,"float32")); s.input_finished()
    while r.is_ready(s): r.decode_stream(s)
    x=r.get_result(s); return (x if isinstance(x,str) else x.text).strip()
def enum_ids(src):
    try:
        out=subprocess.run([YTDLP,"--flat-playlist","--print","%(id)s",src],capture_output=True,text=True,timeout=120,env=ENV).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception as e: print("enum err",src,str(e)[:50],flush=True); return []
def dl_audio(vid,out):
    subprocess.run([YTDLP,"--js-runtimes","deno","-f","140/bestaudio[ext=m4a]/bestaudio",
        "--match-filter","!is_live & duration>120 & duration<14400","-x","--audio-format","wav",
        "--postprocessor-args","ExtractAudio:-ar 16000 -ac 1","--no-playlist","-q",
        "-o",out,f"https://www.youtube.com/watch?v={vid}"],timeout=900,check=True,env=ENV)

done={os.path.basename(w).split("_")[0] for w in glob.glob("yt_wavs/*.wav")}
ids=[]
for s in SOURCES:
    for v in enum_ids(s):
        if v not in done and v not in ids: ids.append(v)
print(f"enumerated {len(ids)} new videos ({len(done)} done)",flush=True)
lock=threading.Lock(); man=open("yt_manifest.tsv","a",encoding="utf-8")
state={"h":sum(sf.info(w).duration for w in glob.glob('yt_wavs/*.wav'))/3600,"n":len(glob.glob('yt_wavs/*.wav'))}
def work(vid):
    if state["h"]>=TARGET_H: return
    raw=f"yt_wavs/_tmp_{vid}.wav"
    try:
        dl_audio(vid,raw); a,_=sf.read(raw,dtype="float32"); a=a.mean(1) if a.ndim>1 else a
    except Exception:
        for p in glob.glob(f"yt_wavs/_tmp_{vid}*"):
            try: os.remove(p)
            except: pass
        return
    W=int(WIN*16000); rows=[]
    for k in range(0,len(a),W):
        seg=a[k:k+W]
        if len(seg)<16000 or float(np.abs(seg).max())<0.01: continue
        txt=asr(seg); zh=[c for c in txt if CJK.match(c)]
        if len(zh)<3: continue
        sp=f"yt_wavs/{vid}_{k//W}.wav"; sf.write(sp,seg,16000); rows.append((sp,t2s.convert(txt),len(seg)/16000))
    try: os.remove(raw)
    except: pass
    with lock:
        for sp,txt,d in rows: man.write(f"{os.path.abspath(sp)}\t{txt}\n"); state["h"]+=d/3600; state["n"]+=1
        man.flush()
        if rows: print(f"  +{vid} ({len(rows)} segs) -> {state['n']} segs, {state['h']:.1f}h",flush=True)
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs=[ex.submit(work,v) for v in ids]
    for _ in as_completed(futs):
        if state["h"]>=TARGET_H: break
print(f"DONE: {state['n']} segments, {state['h']:.1f}h Taiwan YouTube -> yt_manifest.tsv",flush=True)
