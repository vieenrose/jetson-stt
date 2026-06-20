# Datasets — zh-TW / en code-switch for fine-tune and eval

Two uses: **fine-tune** (Tier 3+) and **eval** (every tier). They have different bars — eval needs
*clean, code-switch-rich, Taiwan* references; fine-tune needs *volume* plus a retention set. Audio and
derived manifests are **gitignored**; only text/term seeds live in the repo.

## Taiwan Mandarin (for accent adaptation + TW eval)
| Corpus | What | License / access | Use |
|---|---|---|---|
| **TAT** (Taiwanese Across Taiwan) | read + spontaneous Taiwan Mandarin | research license (apply) | FT + eval |
| **Formosa Speech (FSR-2018/2020)** | Taiwan Mandarin ASR sets | research license | FT |
| **Common Voice zh-TW** | crowd-read Taiwan Mandarin | CC0 | FT + eval (easy to get) |
| **NER-Trs-Vol** | Taiwan broadcast/news | research license | FT |

## zh-en code-switching (for boundary/style coverage)
| Corpus | What | Note |
|---|---|---|
| **ASCEND** | spontaneous zh-en code-switch (HK) | mainland/HK accent — style/boundary coverage, not TW accent |
| **SEAME** | Singapore/Malaysia zh-en code-switch | non-TW accent; valuable for switch-point modeling |
| **NTUML2021 / CSZS** | lecture-style zh-en | academic register |

## Retention set (mandatory for Tier 3)
A frozen sample of the **original training distribution** — English (LibriSpeech/GigaSpeech slice),
mainland zh (AISHELL slice), and zh-en code-switch — mixed into every fine-tune batch to prevent
catastrophic forgetting. This is the direct analog of `jetson-tts`'s `en_retention` set. Size it to
≥30–50% of each batch; without it English WER collapses (the `jetson-tts` accent experiments measured
exactly this failure).

## Eval sets (held out, never trained on)
- A **≥70-utterance zh-TW/en code-switch** dev/test set with **Traditional** references — small sets are
  too noisy to trust (a lesson carried over from `jetson-tts`'s ASR gate).
- Per-slice held-out sets for the MER gate: TW Mandarin (CER), English (WER), mainland zh (CER),
  code-switch (MER + boundary-F1).
- Seed transcripts: [`../data/text/eval.tsv`](../data/text/eval.tsv) — expand before trusting numbers.

## Generating more code-switch data
If labeled TW-accent code-switch audio is scarce (it is), the `jetson-tts` precedent is to **synthesize a
teacher corpus** (e.g. voice-cloned TW-accent TTS) for *offline* augmentation — useful as FT data, with
the caveat that synthetic-accent audio is a weaker teacher than real recordings. Treat synthetic data as
a supplement to TAT/CommonVoice, never the sole source.
