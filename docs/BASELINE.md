# Baseline — X-ASR int8 on sherpa-onnx CPU, characterized on a real Nano

The selected STT engine for the attendant: **X-ASR zh-en streaming zipformer2 transducer**
([`GilgameshWind/X-ASR-zh-en`](https://huggingface.co/GilgameshWind/X-ASR-zh-en)), **int8**, run on
**sherpa-onnx, CPU only**. This doc records *why* — the measured numbers and the methodology — so the
choice is auditable and re-runnable with `scripts/bench_nano.py`.

## Model
- **Type**: streaming zipformer2 transducer (RNN-T), k2-fsa / icefall export, with punctuation.
- **Architecture** (identical across chunk variants): 6 stacks / 19 layers, encoder dims
  192·256·512·768·512·256, vocab 5000 (CJK chars + English BPE), 16 kHz input, 80-d fbank.
- **Streaming variants**: one offline-streaming-unified model exported at chunk shift **160 / 480 / 960 /
  1920 ms** (encoder T = 29 / 61 / 109 / 205). Latency vs. accuracy trade-off; the **960 ms** variant is
  the characterization default (balanced), **480 ms** is the low-latency option for the attendant's
  barge-in responsiveness.

## Why X-ASR at all
Among **every streaming model we could run real-time on the Nano**, X-ASR holds the **record accuracy on
zh/en code-switching**. The baseline decision is an accuracy bake-off result, not a convenience pick —
the rest of this project pushes an already-winning model toward zh-TW rather than swapping engines.

## Test conditions
- **Device**: Jetson Nano gen1, Tegra X1 / GM20B (sm_53), L4T R32.5.1, CUDA 10.2, MAXN power mode.
  `jetson_clocks` **not** pinned → expect ~10% run-to-run variance.
- **Runtime**: sherpa-onnx, onnxruntime **1.6.0** (the Nano's stuck-at version).
- **Clip**: 10 s zh-en code-switch reference — `昨天是 Monday，today is 礼拜二，the day after tomorrow 是星期三`
  (960 ms variant: 1105 frames, 11 streaming chunks). Steady-state, warm-up excluded.
- **Metric**: **encoder ms/chunk** — the fair cross-engine number. (sherpa runs the RNN-T greedy loop in
  Python, which inflates *total* ms vs C++ engines; encoder time is pure optimized inference.)

## Result — thread scaling, encoder ms/chunk (960 ms variant)

| threads | **int8** | fp32 | real-time? |
|--------:|---------:|-----:|:----------:|
| 1 | 562.9 | 649.5 | ✓ |
| **2** | **378.8** | 444.7 | ✓ — **RTF ≈ 0.39** (378.8 ms work per 960 ms audio) |
| 3 | 330.3 | 396.4 | ✓ |
| 4 | 329.8 | 396.9 | ✓ |

### Findings
- **int8 wins** and is **transcription-lossless** vs fp32 on the code-switch clip.
- **Saturates at 3 threads** — the 4th core adds nothing (onnxruntime/MLAS caps useful parallelism here).
  For the attendant this is a *feature*: STT's useful envelope is ≤3 cores, so pinning it to **2 leaves
  ≥2 cores for TTS + I/O** at a ~13% latency cost vs 3 threads (378.8 → 330.3 ms).
- **2 threads is the operating point**: RTF ≈ 0.39 means ~2.5× real-time headroom — enough to absorb the
  Python greedy overhead and co-tenant scheduling jitter and still stay real-time.

## Why not CUDA on this model
- **sherpa-onnx CUDA is *not runnable*** on the Nano's onnxruntime 1.6.0 for this graph.
- The **ggml/CUDA (sm_53) offload path** lives in
  [`x-asr-rapidspeech`](https://github.com/vieenrose/x-asr-rapidspeech): on the real Nano it needs all 4
  CPU cores to *approach* sherpa int8 and its best CUDA encoder time (~355 ms, q8_0) does **not** beat
  int8 sherpa at 2 threads while also contending for the shared GPU that the attendant may want for other
  work. **CPU int8 sherpa is the better fit under a shared 2-core budget.** Full cross-engine tables:
  `x-asr-rapidspeech/BENCHMARKS.md` §4.

## Reproduce
```bash
python scripts/bench_nano.py \
    --model-dir /path/to/x-asr/960ms \
    --int8 \
    --threads 1 2 3 4 \
    --wav data/audio/codeswitch_ref.wav
```
Reports encoder ms/chunk, end-to-end RTF, and first-partial latency per thread count.
