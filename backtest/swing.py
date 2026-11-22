#!/usr/bin/env python3
"""
================================================================
   EGX Swing Backtest - محفظة حقيقية بقواعد دخول وخروج صريحة
================================================================
بدل "ترتيب أسهم كل يوم"، ده بيختبر طريقة تداول كاملة زي ما هتنفذها فعلاً:
  1) درجات الموديل walk-forward (LightGBM على 20/40/60 يوم + Alpha158، بيتدرّب على الماضي بس،
     ويتعاد تدريبه كل 63 يوم) → متوسط ترتيب الـ3.
  2) قواعد swing_core: دخول بافتتاح اليوم التالي، وقف متحرك بالتذبذب، خروج بالترتيب/المدة/السوق.
  3) اختيار الإعدادات على فترة التطوير (2012-2021) بس، والحكم على 2022+ اللي الإعدادات مشافتهاش.
  4) مقارنة بنفس القواعد على درجات بديلة (عشوائي / تذبذب منخفض) عشان نعرف الموديل نفسه بيضيف ولا لأ.

الاستخدام:
  python backtest/swing.py                      # بيبني الدرجات (أو بيقرا الكاش) وبيعمل الاختبار
  python backtest/swing.py --rebuild            # يعيد تدريب الدرجات
  python backtest/swing.py --source synthetic-planted   # اختبار سلامة سريع
================================================================
"""
import argparse
import itertools
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as bt  # noqa: E402
import swing_core as sc  # noqa: E402

DEV = (pd.Timestamp("2012-01-01"), pd.Timestamp("2022-01-01"))
TEST = (pd.Timestamp("2022-01-01"), pd.Timestamp("2100-01-01"))
PRED_FILE = os.path.join(bt.RES_DIR, "swing_preds.npz")
BEST_FILE = os.path.join(bt.RES_DIR, "swing_best.json")
HORIZONS = (20, 40, 60)

GRID = dict(q_in=[0.90, 0.95], q_out=[0.30, 0.50], max_pos=[3, 5, 8], trail_k=[0.0, 2.5, 3.5],
            max_hold=[40, 60], regime=[0, 100, 200], trend50=[False, True], exit_on_regime=[False, True])


def robust_pick(results, keys):
    """اختيار ضد الـoverfitting (على التطوير بس): درجة كل إعداد = متوسط Sharpe بتاعه وبتاع كل الإعدادات اللي
    بتختلف عنه في قيمة واحدة بس. كده بنختار منطقة كويسة مش نقطة محظوظة."""
    idx = {tuple(r[0][k] for k in keys): r for r in results}
    best, best_s = None, -1e9
    for kv, r in idx.items():
        if not r[1] or r[1]["n"] < 80:
            continue
        nb = [r[1]["sharpe"]]
        for j, k in enumerate(keys):
            for v in GRID[k]:
                if v == kv[j]:
                    continue
                q = idx.get(kv[:j] + (v,) + kv[j + 1:])
                if q and q[1]:
                    nb.append(q[1]["sharpe"])
        sc_ = float(np.mean(nb))
        if sc_ > best_s:
            best, best_s = r, sc_
    return best, best_s


def load_data(a):
    if a.source == "yfinance":
        panel = bt.fetch_yfinance(bt.load_tickers())
    else:
        import backtest_horizons as bh
        panel = bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    _f, _m, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, a.min_turnover)
    return C, H, L, V, O, liquid, lock


def features(C, H, L, V, O, liquid, lock, a):
    """مصفوفة المدخلات (T,N,K) ومين صالح: المدخلات الأساسية + EGX30/الدولار + Alpha158."""
    import alpha as al
    import improve as im
    import macro
    _f, mkt0, *_ = bt.build_features(C, H, L, V, O, a.min_turnover)
    if a.source == "yfinance":
        M = macro.fetch_macro(fresh=True)
    else:
        M = macro.synthetic_macro(mkt0["mkt1"])
    M, _src = macro.ensure_egx30(M, C, V)
    X, _ok = im.setup(C, H, L, V, O, a.min_turnover, M)
    XA, _names = al.build_alpha(C, H, L, V, O, liquid, bt.csrank)
    X2 = np.concatenate([X, XA], axis=2)
    ok2 = np.isfinite(X2).all(axis=2) & liquid.values & ~lock.values
    return X2, ok2


def rank_mean(preds):
    st = np.stack([pd.DataFrame(p).rank(axis=1, pct=True).values for p in preds])
    S = np.where(np.isfinite(st).all(axis=0), st.mean(axis=0), np.nan)
    return pd.DataFrame(S).rank(axis=1, pct=True).values     # نرجّعها توزيع 0..1 تاني


def build_scores(C, H, L, V, O, liquid, lock, a):
    """درجات walk-forward: ترتيب متوسط LightGBM 20/40/60 (نفس اللي في final_test)."""
    import improve2 as i2
    X2, ok2 = features(C, H, L, V, O, liquid, lock, a)
    preds = []
    for h in HORIZONS:
        t = time.time()
        TGT, E, fv, Y = i2.prep(C, O, h, ok2)
        p, _t0 = i2.walk("lgbm", X2, Y, E, fv, a, h - 1)
        print(f"[swing] درجات h={h} خلصت في {time.time() - t:.0f}s", flush=True)
        preds.append(np.where(ok2, p, np.nan))
    return rank_mean(preds)


def alt_scores(C, liquid, kind, seed=0):
    if kind == "random":
        rng = np.random.default_rng(seed)
        R = pd.DataFrame(rng.random(C.shape), index=C.index, columns=C.columns)
        R = R.rolling(20, min_periods=1).mean()            # عشوائي بس ثابت نسبيًا (عشان ميعملش churn)
    else:  # lowvol
        r = (C / C.shift(1) - 1).where(lambda x: x.abs() <= 0.25)
        R = -r.rolling(60, min_periods=40).std()
    return R.where(liquid).rank(axis=1, pct=True).values


def fmt(m):
    if not m:
        return "| - | - | - | - | - | - | - | - | - |"
    return (f"| {m['cagr']:+.1f}% | {m['mkt_cagr']:+.1f}% | {m['mdd']:.1f}% ({m['mkt_mdd']:.0f}%) | {m['sharpe']:.2f} | "
            f"{m['n']} | {m['win']:.0f}% | {m['avg_win']:+.1f}% / {m['avg_loss']:+.1f}% | {m['pf']:.2f} | "
            f"{m['hold']:.0f} | {m['expo']:.0f}% |")


HEAD = ["| عائد سنوي | السوق سنوي | أقصى هبوط (السوق) | Sharpe | صفقات | كسبانة | متوسط مكسب / خسارة | Profit factor | أيام احتفاظ | وقت في السوق |",
        "|---|---|---|---|---|---|---|---|---|---|"]


def pdesc(p):
    return (f"أعلى {round((1 - p['q_in']) * 100)}% دخول، خروج تحت {p['q_out']:.0%}، {p['max_pos']} مراكز، "
            f"وقف {p['trail_k']}×ATR{'' if p['trail_k'] else ' (من غير)'}، أقصى {p['max_hold']} يوم، "
            f"فلتر السوق {('SMA' + str(p['regime'])) if p['regime'] else 'مفيش'}، فوق SMA50: {'آه' if p['trend50'] else 'لأ'}، "
            f"خروج لما السوق يقلب: {'آه' if p['exit_on_regime'] else 'لأ'}")


def run(a):
    C, H, L, V, O, liquid, lock = load_data(a)
    T, N = C.shape
    cache_ok = (not a.rebuild and a.source == "yfinance" and os.path.exists(PRED_FILE))
    S = None
    if cache_ok:
        z = np.load(PRED_FILE, allow_pickle=True)
        d0 = pd.DatetimeIndex(z["dates"])
        Sc = pd.DataFrame(z["S"].astype(float), index=d0, columns=list(z["cols"]))
        S = Sc.reindex(index=C.index, columns=C.columns).values
        print(f"[swing] درجات من الكاش لحد {d0[-1].date()}")
    if S is None:
        S = build_scores(C, H, L, V, O, liquid, lock, a)
        if a.source == "yfinance":
            os.makedirs(bt.RES_DIR, exist_ok=True)
            np.savez_compressed(PRED_FILE, S=S.astype(np.float16), dates=np.array(C.index.astype(str)),
                                cols=np.array(C.columns))
    first = int(np.where(np.isfinite(S).sum(axis=1) >= 30)[0][0])
    D = sc.prepare(C, H, L, V, O, S, liquid, lock)
    dev = DEV if a.source == "yfinance" else (C.index[first], C.index[first + (T - first) * 2 // 3])
    test = TEST if a.source == "yfinance" else (dev[1], pd.Timestamp("2100-01-01"))
    print(f"[swing] أول يوم بدرجات: {C.index[first].date()} | تطوير {dev[0].date()}→{dev[1].date()} | اختبار {test[0].date()}+")

    keys = list(GRID)
    results = []
    t_start = time.time()
    for vals in itertools.product(*[GRID[k] for k in keys]):
        p = dict(sc.DEFAULT, **dict(zip(keys, vals)), cost=a.cost_bps / 1e4)
        eq, ex, tr = sc.simulate(D, p, first, T)
        md, mt = sc.metrics(D, eq, ex, tr, *dev), sc.metrics(D, eq, ex, tr, *test)
        results.append((p, md, mt))
    print(f"[swing] {len(results)} إعداد في {time.time() - t_start:.0f}s")
    ok = [r for r in results if r[1] and r[1]["n"] >= 80]
    ok.sort(key=lambda r: -r[1]["sharpe"])
    top1 = ok[0]
    (best_p, best_dev, best_test), nb_sharpe = robust_pick(results, keys)

    # الأفضل + ثباته: متوسط الـ10 الأوائل على التطوير، وإيه اللي حصل لهم في الاختبار
    top10 = ok[:10]
    eq, ex, tr = sc.simulate(D, best_p, first, T)
    yr = sc.yearly(D, eq, C.index[first])
    # تكلفة أعلى
    p35 = dict(best_p, cost=(a.cost_bps + 15) / 1e4)
    eq35, ex35, tr35 = sc.simulate(D, p35, first, T)
    pdl = dict(best_p, exec_close=True)
    eqd, exd, trd = sc.simulate(D, pdl, first, T)
    # نفس القواعد بدرجات بديلة
    alts = {}
    for kind in ("random", "lowvol"):
        D2 = sc.prepare(C, H, L, V, O, alt_scores(C, liquid, kind), liquid, lock)
        e2, x2, t2 = sc.simulate(D2, best_p, first, T)
        alts[kind] = (sc.metrics(D2, e2, x2, t2, *dev), sc.metrics(D2, e2, x2, t2, *test))
    # من غير الموديل خالص: نفس القواعد بس الموديل شايل (فلتر السوق + SMA50 + low-vol)
    reasons = pd.Series([x["why"] for x in tr if test[0] <= C.index[x["et"]]]).value_counts()

    out = [f"# EGX Swing Backtest - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- الأسهم {N} | {C.index[0].date()} → {C.index[-1].date()} | تكلفة {a.cost_bps:.0f} bps في كل اتجاه | "
           f"شراء وبيع بافتتاح الجلسة اللي بعد الإشارة | السوق = متوسط الأسهم السايلة (equal-weight)",
           f"- الإعدادات اتختارت على فترة التطوير {dev[0].year}-{dev[1].year - 1} بس: الإعداد اللي هو وجيرانه (اللي بيختلفوا في قيمة واحدة) "
           f"أعلى متوسط Sharpe ({nb_sharpe:.2f}) من {len(ok)} إعداد، عشان منختارش نقطة محظوظة. "
           f"**فترة الاختبار {test[0].year}+ الموديل والإعدادات مشافوهاش.**", "",
           "## الإعداد المختار", "", pdesc(best_p), "",
           "| الفترة " + HEAD[0], "|---" + HEAD[1],
           f"| تطوير {dev[0].year}-{dev[1].year - 1} " + fmt(best_dev),
           f"| **اختبار {test[0].year}+** " + fmt(best_test),
           f"| اختبار بتكلفة {a.cost_bps + 15:.0f} bps " + fmt(sc.metrics(D, eq35, ex35, tr35, *test)),
           "| اختبار بتنفيذ بقفلة اليوم التالي (تأخير) " + fmt(sc.metrics(D, eqd, exd, trd, *test)), "",
           f"للمقارنة: أعلى Sharpe منفرد على التطوير كان ({pdesc(top1[0])}) → اختبار:", "", "|" + HEAD[0][1:], HEAD[1],
           fmt(top1[2]), "",
           "## هل الموديل نفسه بيضيف؟ (نفس القواعد بالظبط بدرجات تانية)", "",
           "| الدرجات " + HEAD[0], "|---" + HEAD[1],
           "| الموديل - اختبار " + fmt(best_test)]
    for kind, lab in (("random", "عشوائي"), ("lowvol", "تذبذب منخفض")):
        out.append(f"| {lab} - اختبار " + fmt(alts[kind][1]))
    out += ["", "## سنة بسنة (الإعداد المختار)", "", "| السنة | المحفظة | السوق | الفرق |", "|---|---|---|---|"]
    out += [f"| {y} | {s:+.1f}% | {m:+.1f}% | {s - m:+.1f}% |" for y, s, m in yr]
    out += ["", "## أسباب الخروج في فترة الاختبار", ""]
    out += [f"- {sc.REASON_AR.get(k, k)}: {v}" for k, v in reasons.items()]
    out += ["", "## ثبات النتيجة: أعلى 10 إعدادات على التطوير وأداهم في الاختبار", "",
            "| # | الإعداد | Sharpe تطوير | عائد سنوي اختبار | السوق | أقصى هبوط اختبار | Sharpe اختبار |", "|---|---|---|---|---|---|---|"]
    for k, (p, md, mt) in enumerate(top10, 1):
        out.append(f"| {k} | {pdesc(p)} | {md['sharpe']:.2f} | {mt['cagr']:+.1f}% | {mt['mkt_cagr']:+.1f}% | {mt['mdd']:.1f}% | {mt['sharpe']:.2f} |"
                   if mt else f"| {k} | {pdesc(p)} | {md['sharpe']:.2f} | - | - | - | - |")
    tests = [r[2] for r in ok if r[2]]
    beat = sum(1 for m in tests if m["cagr"] > m["mkt_cagr"])
    out += ["", f"من كل الـ{len(tests)} إعداد: {beat} غلبوا السوق في فترة الاختبار "
            f"({beat / max(len(tests), 1) * 100:.0f}%)، ومتوسط فرق العائد السنوي "
            f"{np.mean([m['cagr'] - m['mkt_cagr'] for m in tests]):+.1f}%.", ""]
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "latest" if a.source == "yfinance" else a.source
    with open(os.path.join(bt.RES_DIR, f"report_swing_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    if a.source == "yfinance":
        with open(BEST_FILE, "w", encoding="utf-8") as fh:
            json.dump({"params": best_p, "dev": best_dev, "test": best_test, "made": f"{datetime.now():%Y-%m-%d %H:%M}",
                       "data_to": str(C.index[-1].date())}, fh, ensure_ascii=False, indent=1)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--min-turnover", type=float, default=1_000_000)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    try:
        run(a)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        with open(os.path.join(bt.RES_DIR, "error_swing_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
