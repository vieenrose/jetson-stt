---
title: X-ASR zh-TW/en — Original vs Fine-tuned
emoji: 🎙️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
---

# X-ASR zh-TW/en — Original vs Fine-tuned (streaming, Jetson-Nano-class)

Side-by-side compare of the **original X-ASR** streaming zipformer (Simplified output) vs the
**[`jetson-stt`](https://github.com/vieenrose/jetson-stt)** fine-tune that outputs **native Traditional
zh-TW + lowercase English** directly. Includes zh-TW / English code-switch demo samples (NTU ML lectures).
Also shows the zero-retrain OpenCC `s2twp` Taiwan post-processing for the original model. Research
demonstration — see the linked model cards for caveats (in-domain fine-tune).
