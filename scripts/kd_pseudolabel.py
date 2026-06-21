"""Phase 1 of half-size distillation: pseudo-label training audio with the TEACHER (deployed 480ms int8)."""
import os, io, csv, glob, tarfile
import numpy as np, soundfile as sf, librosa, sherpa_onnx

DEP = "/tmp/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"
N_CV = int(os.environ.get("N_CV", "4000"))
os.makedirs("kd_wavs", exist_ok=True)
rec = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=f"{DEP}/tokens.txt", encoder=f"{DEP}/encoder.int8.onnx", decoder=f"{DEP}/decoder.onnx",
    joiner=f"{DEP}/joiner.int8.onnx", num_threads=4, provider="cpu", decoding_method="greedy_search",
    sample_rate=16000, feature_dim=80)
def teach(a):
    s = rec.create_stream(); s.accept_waveform(16000, a.astype("float32"))
    s.accept_waveform(16000, np.zeros(32000, "float32")); s.input_finished()
    while rec.is_ready(s): rec.decode_stream(s)
    x = rec.get_result(s); return (x if isinstance(x, str) else x.text).strip()

man = open("kd_manifest.tsv", "w", encoding="utf-8"); n = 0
# NTUML (wavs on disk)
for w in sorted(glob.glob("ftdata/tw_train/*.wav")):
    a, sr = sf.read(w, dtype="float32"); a = a.mean(1) if a.ndim > 1 else a
    t = teach(a)
    if t: man.write(f"{os.path.abspath(w)}\t{t}\n"); n += 1
print(f"NTUML pseudo-labeled: {n}", flush=True)
# CV zh-TW train (extract mp3 -> wav, pseudo-label)
TAR = open("cv_train_tar_path.txt").read().strip()
nc = 0
with tarfile.open(TAR) as tar:
    for m in tar:
        if nc >= N_CV: break
        if not m.name.endswith(".mp3"): continue
        try: a, _ = librosa.load(io.BytesIO(tar.extractfile(m).read()), sr=16000, mono=True)
        except Exception: continue
        if len(a) < 1600: continue
        t = teach(a)
        if not t: continue
        wp = os.path.abspath(f"kd_wavs/{os.path.basename(m.name)[:-4]}.wav")
        sf.write(wp, a, 16000); man.write(f"{wp}\t{t}\n"); nc += 1
        if nc % 500 == 0: print(f"CV pseudo-labeled: {nc}/{N_CV}", flush=True)
man.close()
print(f"DONE: {n} NTUML + {nc} CV = {n+nc} pseudo-labeled clips -> kd_manifest.tsv", flush=True)
