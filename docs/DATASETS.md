# Datasets — zh-TW / en code-switch for fine-tune and eval

Two uses: **fine-tune** (Tier 3+) and **eval** (every tier). They have different bars — eval needs
*clean, code-switch-rich, Taiwan* references; fine-tune needs *volume* plus a retention set. Audio and
derived manifests are **gitignored**; only text/term seeds live in the repo.

> **Verified by research ([`RESEARCH.md`](RESEARCH.md)). Two corrections from the original draft:**
> (1) **TAT (Taiwanese Across Taiwan) is Taiwanese *Hokkien* (台語), not Taiwan Mandarin** — wrong
> language, removed. (2) **There is no public corpus of authentic Taiwan-accent zh-en code-switch
> audio** — the central bottleneck. Taiwan accent must largely be *manufactured* (synthetic /
> Breeze-pseudo-labeled). Full citations and license caveats in `RESEARCH.md`.

## Taiwan-accent audio (the scarce, valuable axis)
| Corpus | What | Code-switch | Size | License / access | Use |
|---|---|---|---|---|---|
| **NTUML2021** (`ky552/ML2021_ASR_ST`) | NTU ML lectures, TW-Mandarin + en | **yes** | ~11h (5h long eval) | unstated → **verify** · HF direct | **FT + eval** (only free TW-accent CS audio) |
| **Common Voice zh-TW** | crowd-read TW Mandarin | no | ~80h val (131h tot) | **CC0** · direct | FT + eval (only large free TW-Mandarin) |
| NER-Trs-Vol | Taiwan-Mandarin radio (the real Formosa set) | no | ~610h | **Non-commercial** · FSW/ACLCLP app | FT — *NC blocks product use* |
| MATBN | Taiwan-Mandarin broadcast news | no | 198h | ACLCLP, NC · app | FT + eval |

## zh-en code-switch — mainland/HK/SEA accent (switch-point coverage + retention, NOT TW accent)
| Corpus | What | Size | License / access |
|---|---|---|---|
| **TAL-CSASR** (`csukuangfj/tal_csasr`) | mainland zh-en (baseline's own CS training data) | ~587h | unstated → verify · speechhome/HF |
| **ASCEND** (`CAiRE/ASCEND`) | HK spontaneous zh-en | 10.6h | CC-BY-SA 4.0 · HF |
| **CS-Dialogue** (`BAAI/CS-Dialogue`) | mainland zh-en dialogue | 104h | CC-BY-**NC**-SA · HF |
| ASRU2019 CS | mainland zh-en | 240h | DataTang · challenge reg. |
| SEAME | Singapore/Malaysia zh-en | ~30–192h | **LDC (paid)** |
| **CSZS-zh-en** (`ky552/cszs_zh_en`) | CS eval benchmark | 35k utt | **MIT** · HF |

## Retention set (mandatory for Tier 3) — mirror the baseline's upstream recipe
The X-ASR lineage (`multi_zh_en` zipformer) trained on **LibriSpeech 960h (en) + TAL-CSASR (CS) +
AISHELL-2 (mainland zh)** — so the retention mix should sample those three (free proxies: LibriSpeech +
ASCEND + **AISHELL-1**, Apache-2.0). Mix into every fine-tune batch (icefall `--use-mux` does this if you
have the cuts). Replay evidence: **10–20% original data prevents forgetting; >20% starts a trade-off**
(arXiv 2307.07280). Without it English WER collapses (measured in `jetson-tts` and the Whisper zh-en CS
literature).

## Eval sets (held out, never trained on)
- A **≥70-utterance zh-TW/en code-switch** dev/test set with **Traditional** references — small sets are
  too noisy to trust (a lesson carried over from `jetson-tts`'s ASR gate).
- Per-slice held-out sets for the MER gate: TW Mandarin (CER), English (WER), mainland zh (CER),
  code-switch (MER + boundary-F1).
- Seed transcripts: [`../data/text/eval.tsv`](../data/text/eval.tsv) — expand before trusting numbers.

## Manufacturing Taiwan-accent CS data (since it doesn't exist publicly)
Two validated routes, used together:
1. **Distill Breeze-ASR-25 pseudo-labels.** MediaTek **Breeze-ASR-25** (Whisper-large-v2, Taiwan SOTA,
   zh-TW CER 7.97%, Apache-2.0) is too big for the Nano but an excellent **offline teacher**: pseudo-label
   real Taiwan audio (Common Voice zh-TW, TW podcasts) with confidence filtering and fine-tune the X-ASR
   streaming student. Validated by "streaming transducer from Whisper pseudo-labels" (arXiv 2409.13499).
2. **TTS synthesis** (the `jetson-tts` precedent): BreezyVoice / `matcha-zh-tw-en` to render TW-accent
   zh-en CS audio for offline augmentation. Breeze-ASR-25 itself trained on **10,000h synthetic Chinese**
   — proof this scales. Treat synthetic as a *supplement* to real NTUML2021 + CV-zh-TW, never the sole
   source.

## Eval suite — reuse Breeze-ASR-25's for comparability
Adopt the published Taiwan SOTA's eval suite so numbers are comparable: **ASCEND-{OVERALL/EN/ZH/MIX},
Common Voice zh-TW, CSZS-zh-en, ML-Lecture-2021-Long** (+ Formosa-* long-form if obtainable). Plus the
project's own per-slice MER gate (TW-CER / EN-WER / mainland-CER / CS-MER + boundary-F1) and the
`s2twp`-isolation test (`EVAL.md`) to separate orthographic from acoustic error. Use **≥70 utterances**
per slice (the `jetson-tts` lesson). Seed: [`../data/text/eval.tsv`](../data/text/eval.tsv).

## License caveat (the attendant is a product)
NER/MATBN/CS-Dialogue are non-commercial; NTUML2021 and TAL-CSASR HF cards have **empty license fields**.
The commercially-clean Taiwan sources are **CC0 Common Voice zh-TW + synthetic/pseudo-labeled audio**.
Verify every license before shipping fine-tune data.
