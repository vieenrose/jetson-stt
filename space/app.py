#!/usr/bin/env python3
"""X-ASR zh-TW/en — Simplified vs +OpenCC-s2twp vs native-Traditional, all from ONE deployed model."""
import os, threading, numpy as np, soundfile as sf, gradio as gr
from huggingface_hub import snapshot_download
import sherpa_onnx
from opencc import OpenCC

s2twp = OpenCC("s2twp")
REPO = "Luigi/x-asr-zh-tw-en-streaming-native-demo"  # deployed 480ms int8 model; two tokenizers in-repo

def _rec(d, tokens):
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/{tokens}", encoder=f"{d}/encoder.int8.onnx",
        decoder=f"{d}/decoder.onnx", joiner=f"{d}/joiner.int8.onnx",
        num_threads=2, provider="cpu", decoding_method="greedy_search",
        sample_rate=16000, feature_dim=80)

_R = {}
_LOCK = threading.Lock()
def recs():  # lazy + thread-safe; pre-warmed in the background at boot so the first transcription isn't slow
    if not _R:
        with _LOCK:
            if not _R:
                d = snapshot_download(REPO)
                _R["simp"] = _rec(d, "tokens_simplified.txt")  # original deployed tokenizer -> Simplified
                _R["native"] = _rec(d, "tokens.txt")           # s2twp-relabeled tokenizer  -> Traditional
    return _R

def decode(rec, samples, sr):
    s = rec.create_stream()
    s.accept_waveform(sr, samples)
    s.accept_waveform(sr, np.zeros(int(2.0 * sr), dtype="float32"))  # flush streaming chunk
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    r = rec.get_result(s)
    return (r if isinstance(r, str) else r.text).strip()

def transcribe(audio):
    if audio is None:
        return "", "", ""
    sr, data = audio
    data = data.astype("float32")
    if data.ndim > 1:
        data = data.mean(1)
    if np.abs(data).max() > 1.0:
        data = data / 32768.0
    if sr != 16000:
        import scipy.signal as ss
        data = ss.resample(data, int(len(data) * 16000 / sr)).astype("float32"); sr = 16000
    R = recs()
    simp = decode(R["simp"], data, sr)
    native = decode(R["native"], data, sr)
    return simp, s2twp.convert(simp), native

DESC = """
# X-ASR zh-TW/en — deployed model: Simplified vs OpenCC vs native Traditional

All three outputs come from the **same deployed X-ASR streaming model** (480 ms int8, runs on a Jetson Nano at
2 CPU threads). The point: a **native-Traditional tokenizer matches `deployed + OpenCC s2twp` with no runtime
post-processing**.

- **X-ASR (Simplified)** — what the model emits natively.
- **+ OpenCC `s2twp` (Traditional)** — the zero-retrain Taiwan post-processing step (Tier-1).
- **Native Traditional (no post-step)** — same model with an `s2twp`-relabeled tokenizer; emits Traditional
  directly. On 500 Common Voice zh-TW clips: **0.0675 CER vs 0.0683 for deployed + s2twp** — tied, no OpenCC at runtime.

Model: [`Luigi/x-asr-zh-tw-en-streaming-native-demo`](https://huggingface.co/Luigi/x-asr-zh-tw-en-streaming-native-demo) ·
recipe + benchmarks: [github.com/vieenrose/jetson-stt](https://github.com/vieenrose/jetson-stt).
"""

with gr.Blocks(title="X-ASR zh-TW/en — native Traditional vs OpenCC") as demo:
    gr.Markdown(DESC)
    inp = gr.Audio(sources=["upload", "microphone"], type="numpy", label="zh-TW / English speech")
    btn = gr.Button("Transcribe", variant="primary")
    with gr.Row():
        o1 = gr.Textbox(label="X-ASR (Simplified)")
        o2 = gr.Textbox(label="+ OpenCC s2twp (Traditional, post-step)")
        o3 = gr.Textbox(label="Native Traditional (no post-step)")
    btn.click(transcribe, inputs=inp, outputs=[o1, o2, o3])
    ex = [[f"samples/{f}"] for f in sorted(os.listdir("samples")) if f.endswith(".wav")] if os.path.isdir("samples") else []
    if ex:
        gr.Examples(examples=ex, inputs=inp, label="zh-TW / en code-switch samples (NTU ML lectures)")

if __name__ == "__main__":
    threading.Thread(target=recs, daemon=True).start()  # pre-warm models in background (instant boot, fast first run)
    demo.launch()
