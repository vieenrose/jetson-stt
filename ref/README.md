# ref/ — provenance & upstream pointers

This repo redistributes **no model weights**. Pointers to the upstream artifacts and the sibling repos
that already carry the reference material.

## Base model
- **X-ASR zh-en streaming zipformer2 transducer** — [`GilgameshWind/X-ASR-zh-en`](https://huggingface.co/GilgameshWind/X-ASR-zh-en)
  (k2-fsa / icefall export, sherpa-onnx; Apache-2.0). Mandarin–English code-switching with punctuation;
  4 streaming chunk variants (160/480/960/1920 ms); vocab 5000; 16 kHz / 80-d fbank.
- Holds the **record zh/en code-switch accuracy** among streaming models we could run real-time on the
  Nano — see `../docs/BASELINE.md`.

## Reference material lives in the sibling repo
[`x-asr-rapidspeech`](https://github.com/vieenrose/x-asr-rapidspeech) already vendors the useful X-ASR
reference under its `ref/`:
- `tokens.txt` (5000-token vocab),
- `config.json` (chunk sizes, sample rate, streaming flags),
- `sherpa_streaming_infer.py` (the streaming inference frontend — our CJK-spacing normalization in
  `scripts/zh_tw_postproc.py` mirrors it),
- `icefall/` (the relevant `zipformer.py` / `scaling.py` / `subsampling.py` modules),
- `BENCHMARKS.md` (the on-device Nano numbers cited in `../docs/BASELINE.md`).

Clone it alongside this repo when you need the tokenizer or to inspect the graph:
```bash
git clone https://github.com/vieenrose/x-asr-rapidspeech
```

## Sibling / related repos
- [`jetson-tts`](https://github.com/vieenrose/jetson-tts) — symmetric TTS half of the attendant
- [`RapidSpeech.cpp`](https://github.com/vieenrose/RapidSpeech.cpp) — ggml streaming-zipformer engine (CPU+CUDA sm_53)
- [`streaming-zipformer-demo`](https://github.com/vieenrose/streaming-zipformer-demo) — multi-engine live mic demo (VAD + hotwords + opencc)
