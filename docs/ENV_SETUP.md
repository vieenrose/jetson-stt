# Environment setup

Two environments: a **host** (x86_64 or the GB10 dev box, with a GPU) for fine-tuning and ONNX export,
and the **Nano** for deployment and on-device benchmarking. They are deliberately different — you never
train on the Nano.

## Deploy / bench env (Jetson Nano gen1, aarch64)

The Nano is stuck on **L4T R32.5.1 / CUDA 10.2 / onnxruntime 1.6.0**. Use the prebuilt aarch64
sherpa-onnx wheel that matches that ORT, or build from the
[`sherpa-onnx`](https://github.com/vieenrose/sherpa-onnx) fork.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # sherpa-onnx, numpy, soundfile, opencc, jiwer
# verify the runtime sees the model
python scripts/bench_nano.py --model-dir /path/to/x-asr/960ms --int8 --threads 2 --wav data/audio/ref.wav
```

- Pin clocks for *stable* benchmarking (not for the budget test): `sudo jetson_clocks`. Leave it
  unpinned to measure realistic attendant conditions.
- Restrict cores to emulate the attendant's shared budget: `taskset -c 0,1 python ...` (2 cores).

## Train / export env (host with GPU)

Fine-tuning (Tiers 3–4 of `TRAINING.md`) uses **icefall + k2 + PyTorch** on the host. The Nano never
sees PyTorch.

```bash
# conda recommended for k2/PyTorch CUDA matching
conda create -n jetson-stt-train python=3.10 && conda activate jetson-stt-train
# install torch + k2 + icefall per https://k2-fsa.github.io/icefall/installation/
pip install -r requirements-train.txt    # lhotse, sentencepiece, onnx, onnxruntime-tools
git clone https://github.com/k2-fsa/icefall third_party/icefall   # gitignored
```

- Fine-tune with the icefall **zipformer streaming** recipe; keep `--causal 1` and the chunk/left-context
  config matching the deployed variant.
- Export with `export-onnx-streaming.py`, then int8-quantize for the Nano (see `TRAINING.md` §Tier 3).

## What's gitignored
Model weights (`*.onnx`, `*.pt`), corpora and rendered audio, training runs, and third-party clones —
all large and reproducible. See `.gitignore`. This repo stores **tooling, docs, eval harness, and seed
text/term lists** only.
