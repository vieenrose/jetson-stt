#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MER / CER / WER + punctuation-F1 + casing + code-switch boundary, with bootstrap CIs.

zh-en code-switch can't be scored by a single WER or CER (docs/EVAL.md). This computes everything from
**one** edit-distance alignment over a unified token stream, so cross-boundary substitutions are counted
exactly once (the earlier version re-aligned per language, which double-counted boundary edits).

Token stream:
  - each CJK character           -> one 'zh' token
  - each [A-Za-z0-9'] run        -> one 'en' token (matched case-insensitively; original casing kept)
  - CJK/ASCII punctuation        -> 'punct' tokens, scored SEPARATELY (punctuation-F1), not in MER
  - whitespace                   -> dropped

Reported (with 95% bootstrap CIs over utterances):
  MER   = (S+D+I) / N_ref over all non-punct tokens
  zhCER = zh errors / zh ref ;  enWER = en errors / en ref   (attributed from the single alignment)
  punctF1 = multiset F1 of punctuation marks (the model ships punctuation; the LLM downstream parses it)
  casing  = fraction of aligned English matches whose case is also exact
  boundary P/R/F1 = en<->zh switch-point detection

Usage:
    python scripts/eval_asr.py --hyp hyp.tsv --ref ref.tsv                 # full report + CIs
    python scripts/eval_asr.py --hyp hyp.tsv --ref ref.tsv --metric boundary
    python scripts/eval_asr.py --selftest                                  # sanity-check the metric
"""
import argparse
import random
import re
import sys
from collections import Counter

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_WORD = re.compile(r"[A-Za-z0-9']+")
_PUNCT = set("，。！？；：、（）《》〈〉【】「」『』“”‘’,.!?;:%)]}([{")


class Tok:
    __slots__ = ("surf", "key", "lang")

    def __init__(self, surf, key, lang):
        self.surf, self.key, self.lang = surf, key, lang


def tokenize(text):
    """-> (content_tokens, punct_marks). content lang in {'zh','en'}; punct kept separately."""
    content, punct = [], []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if _CJK.match(ch):
            content.append(Tok(ch, ch, "zh"))
            i += 1
        elif ch.isalnum() or ch == "'":
            m = _WORD.match(text, i)
            if m:
                w = m.group(0)
                content.append(Tok(w, w.lower(), "en"))
                i = m.end()
            else:
                # non-ASCII alphanumeric (full-width digit, accented/Greek letter): own token
                content.append(Tok(ch, ch.lower(), "en"))
                i += 1
        else:
            if ch in _PUNCT:
                punct.append(ch)
            i += 1
    return content, punct


def align(a, b):
    """Levenshtein over token keys with backtrace. a=hyp, b=ref.
    Returns list of ops: ('match'|'sub'|'del'|'ins', hyp_tok_or_None, ref_tok_or_None)."""
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        ai = a[i - 1].key
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1].key else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    # backtrace
    ops = []
    i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1].key == b[j - 1].key else 1):
            ops.append(("match" if a[i - 1].key == b[j - 1].key else "sub", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("ins", a[i - 1], None))  # hyp has extra token
            i -= 1
        else:
            ops.append(("del", None, b[j - 1]))  # ref token missing from hyp
            j -= 1
    ops.reverse()
    return ops


def score_utt(hyp_text, ref_text):
    """Per-utterance counts from a single alignment."""
    hc, hp = tokenize(hyp_text)
    rc, rp = tokenize(ref_text)
    ops = align(hc, rc)
    err = {"zh": 0, "en": 0}
    ref = {"zh": 0, "en": 0}
    tot_err = 0
    case_match = case_total = 0
    for op, h, r in ops:
        if op == "match":
            ref[r.lang] += 1
            if r.lang == "en":
                case_total += 1
                if h.surf == r.surf:
                    case_match += 1
        elif op == "sub":
            ref[r.lang] += 1
            err[r.lang] += 1
            tot_err += 1
        elif op == "del":
            ref[r.lang] += 1
            err[r.lang] += 1
            tot_err += 1
        else:  # ins -> attribute to hyp token lang
            err[h.lang] += 1
            tot_err += 1
    # punctuation multiset F1 counts
    ch, cr = Counter(hp), Counter(rp)
    p_tp = sum((ch & cr).values())
    p_fp = sum(ch.values()) - p_tp
    p_fn = sum(cr.values()) - p_tp
    return {
        "tot_err": tot_err, "tot_ref": ref["zh"] + ref["en"],
        "zh_err": err["zh"], "zh_ref": ref["zh"], "en_err": err["en"], "en_ref": ref["en"],
        "case_match": case_match, "case_total": case_total,
        "p_tp": p_tp, "p_fp": p_fp, "p_fn": p_fn,
    }


def _rate(num, den):
    return num / den if den else 0.0


def aggregate(per_utt):
    s = {k: sum(u[k] for u in per_utt) for u in [per_utt[0]] for k in per_utt[0]}
    return {
        "MER": _rate(s["tot_err"], s["tot_ref"]),
        "zhCER": _rate(s["zh_err"], s["zh_ref"]),
        "enWER": _rate(s["en_err"], s["en_ref"]),
        "casing": _rate(s["case_match"], s["case_total"]),
        "punctF1": _rate(2 * s["p_tp"], 2 * s["p_tp"] + s["p_fp"] + s["p_fn"]),
        "_sums": s,
    }


def bootstrap_ci(per_utt, key_fn, B=1000, seed=1234):
    """95% percentile CI of a rate over utterance resampling."""
    rng = random.Random(seed)
    n = len(per_utt)
    vals = []
    for _ in range(B):
        sample = [per_utt[rng.randrange(n)] for _ in range(n)]
        vals.append(key_fn(sample))
    vals.sort()
    lo = vals[int(0.025 * B)]
    hi = vals[int(0.975 * B)]
    return lo, hi


def load(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            uid, _, text = line.partition("\t")
            d[uid] = text
    return d


def boundaries(content):
    bset = set()
    prev = None
    for idx, t in enumerate(content):
        if prev is not None and t.lang != prev:
            bset.add(idx)
        prev = t.lang
    return bset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hyp")
    ap.add_argument("--ref")
    ap.add_argument("--metric", default="mer", choices=["mer", "boundary"])
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap iterations for CIs (0=off)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())
    if not args.hyp or not args.ref:
        ap.error("--hyp and --ref are required (or use --selftest)")

    hyp, ref = load(args.hyp), load(args.ref)
    ids = [u for u in ref if u in hyp]
    missing = [u for u in ref if u not in hyp]
    if missing:
        print(f"[eval] WARNING: {len(missing)} ref ids missing from hyp", file=sys.stderr)
    if not ids:
        raise SystemExit("[eval] no overlapping ids between hyp and ref")

    if args.metric == "boundary":
        tp = fp = fn = 0
        for u in ids:
            hb = boundaries(tokenize(hyp[u])[0])
            rb = boundaries(tokenize(ref[u])[0])
            tp += len(hb & rb); fp += len(hb - rb); fn += len(rb - hb)
        prec, rec = _rate(tp, tp + fp), _rate(tp, tp + fn)
        f1 = _rate(2 * prec * rec, prec + rec)
        print(f"utts={len(ids)}  switch-points: tp={tp} fp={fp} fn={fn}")
        print(f"boundary  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
        return

    per_utt = [score_utt(hyp[u], ref[u]) for u in ids]
    agg = aggregate(per_utt)
    print(f"utts={len(ids)}")

    def line(name, val, ratefn):
        if args.bootstrap:
            lo, hi = bootstrap_ci(per_utt, ratefn, B=args.bootstrap)
            print(f"{name:8s}= {val:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
        else:
            print(f"{name:8s}= {val:.4f}")

    line("MER", agg["MER"], lambda s: _rate(sum(u["tot_err"] for u in s), sum(u["tot_ref"] for u in s)))
    line("zhCER", agg["zhCER"], lambda s: _rate(sum(u["zh_err"] for u in s), sum(u["zh_ref"] for u in s)))
    line("enWER", agg["enWER"], lambda s: _rate(sum(u["en_err"] for u in s), sum(u["en_ref"] for u in s)))
    print(f"punctF1 = {agg['punctF1']:.4f}   (punctuation marks; LLM downstream parses these)")
    print(f"casing  = {agg['casing']:.4f}   (English matches with exact case; {agg['_sums']['case_total']} en matches)")


# ---------------------------------------------------------------------------
def _approx(a, b, eps=1e-9):
    return abs(a - b) < eps


def _selftest():
    bad = 0
    # 1. identical -> zero error, perfect punct/casing
    pu = [score_utt("昨天是 Monday，today is 礼拜二", "昨天是 Monday，today is 礼拜二")]
    a = aggregate(pu)
    for k in ("MER", "zhCER", "enWER"):
        if not _approx(a[k], 0.0):
            print(f"  FAIL identical {k}={a[k]}"); bad += 1
    if not _approx(a["punctF1"], 1.0) or not _approx(a["casing"], 1.0):
        print(f"  FAIL identical punctF1={a['punctF1']} casing={a['casing']}"); bad += 1

    # 2. one CJK sub + one english-word del, single alignment
    #    ref: 我 是 cat   hyp: 我 是   -> 'cat' deleted (en err=1); '是' matches. zhCER 0/2, enWER 1/1
    u = score_utt("我是", "我是 cat")
    if not (u["zh_err"] == 0 and u["zh_ref"] == 2 and u["en_err"] == 1 and u["en_ref"] == 1):
        print(f"  FAIL del-attribution {u}"); bad += 1

    # 3. casing miss: ref 'Monday' hyp 'monday' -> match (case-insensitive) but casing fails
    u = score_utt("monday", "Monday")
    if not (u["en_err"] == 0 and u["case_total"] == 1 and u["case_match"] == 0):
        print(f"  FAIL casing {u}"); bad += 1

    # 4. punctuation missing in hyp
    u = score_utt("你好 world", "你好，world！")
    # ref punct {，！}=2, hyp punct {}=0 -> tp0 fp0 fn2
    if not (u["p_tp"] == 0 and u["p_fn"] == 2 and u["p_fp"] == 0):
        print(f"  FAIL punct {u}"); bad += 1

    # 5. boundary detection: 我(0) 是(1) cat(2) 的(3) -> zh->en at 2, en->zh at 3
    hb = boundaries(tokenize("我是 cat 的")[0])
    if hb != {2, 3}:
        print(f"  FAIL boundary set {hb}"); bad += 1

    print(f"[selftest] {'PASS' if bad == 0 else f'{bad} FAILURES'}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    main()
