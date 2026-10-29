#!/usr/bin/env python3
"""
================================================================
   EGX - تجارب تحسين الموديل (احتفاظ 10 أيام، تكلفة حقيقية)
================================================================
بيجرّب تعديلات على نفس walk-forward وبيقارنهم بصافي bps للصفقة بعد التكلفة (2022+):
  1) حد السيولة: 1 / 3 / 6 مليون جنيه (متوسط تداول يومي)
  2) حجم الصف المختار: أعلى 10% / 20% / 30%
  3) فلتر السوق: متدخلش لما السوق نازل بقوة (وسيط آخر 5 أيام <= -2%)
  4) ensemble: متوسط ترتيب موديلات فترات 5 و10 و20 يوم

الاستخدام:
  python backtest/improve.py --source yfinance
  python backtest/improve.py --source synthetic-planted --trees 30 --block 126
================================================================
"""

import argparse
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as bt  # noqa: E402
import backtest_horizons as bh  # noqa: E402
import diagnose as dg  # noqa: E402
import macro  # noqa: E402

RECENT = 2022


def setup(C, H, L, V, O, min_tov, M):
    f, mkt, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, min_tov)
    lr = np.log1p(f["r1"])
    f["r60"] = np.expm1(lr.rolling(60, min_periods=60).sum())
    f["r120"] = np.expm1(lr.rolling(120, min_periods=120).sum())
    N = C.shape[1]
    Xcs, _names = macro.build_X(f, mkt, liquid, N, bh.XF2, bt.MF, M, bt.csrank)
    feat_ok = np.isfinite(Xcs).all(axis=2) & liquid.values & ~lock.values
    return Xcs, feat_ok


def fit_pred(Xcs, feat_ok, C, O, h, a, first_valid_hint=None):
    TGT = bh.build_target(C, O, h).values
    E = feat_ok & np.isfinite(TGT)
    elig = E.sum(axis=1)
    min_cross = int(np.clip(0.5 * np.percentile(elig[elig > 0], 75), 20, bt.MIN_CROSS))
    first_valid = int(np.where(elig >= min_cross)[0][0])
    Y = (pd.DataFrame(np.where(E, TGT, np.nan)).rank(axis=1, pct=True) - 0.5).values.astype(np.float32)
    Y = np.nan_to_num(Y)
    pred, t0 = dg.walk_forward(Xcs, Y, E, first_valid, a.min_train, a.block, a.trees, h - 1, C.index, halflife=365)
    return pred, t0


def eval_pred(pred, TGT, E, t0, T, dates, cost, frac=0.2, skip=None):
    """صافي bps للصفقة 2022+ (أعلى frac من الأسهم - متوسط السوق). skip: مصفوفة bool لأيام نتجنبها."""
    ex = []
    for t in range(t0, T - 1):
        if dates[t].year < RECENT or (skip is not None and skip[t]):
            continue
        m = E[t] & np.isfinite(pred[t])
        n = int(m.sum())
        if n < 30:
            continue
        cols = np.where(m)[0]
        s, y = pred[t, cols], TGT[t, cols]
        top = cols[np.argsort(-s)[: max(int(n * frac), 5)]]
        ex.append(float(TGT[t, top].mean() - y.mean()))
    if len(ex) < 50:
        return None
    ex = np.array(ex)
    return {"days": len(ex), "gross": ex.mean() * 1e4, "net": (ex.mean() - 2 * cost) * 1e4,
            "t": ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)) + 1e-12),
            "hit": float((ex - 2 * cost > 0).mean() * 100)}


def rank_avg(preds):
    out = np.full_like(preds[0], np.nan)
    rs = [pd.DataFrame(p).rank(axis=1, pct=True).values for p in preds]
    st = np.stack(rs)
    ok = np.isfinite(st).all(axis=0)
    out[ok] = st.mean(axis=0)[ok]
    return out


def row(label, r):
    if not r:
        return f"| {label} | - | - | - | - | - |"
    return f"| {label} | {r['days']} | {r['gross']:+.0f} | {r['net']:+.0f} | {r['t']:+.1f} | {r['hit']:.0f}% |"


def run(a):
    cost = a.cost_bps / 1e4
    if a.source == "yfinance":
        panel = bt.fetch_yfinance(bt.load_tickers())
    else:
        panel = bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    dates, (T, N) = C.index, C.shape
    mk5 = (C / C.shift(5) - 1).clip(-0.6, 0.6).median(axis=1).values
    f0, mkt0, *_ = bt.build_features(C, H, L, V, O, 1_000_000)
    M = macro.fetch_macro(fresh=True) if a.source == "yfinance" else macro.synthetic_macro(mkt0["mkt1"])
    M, egx_src = macro.ensure_egx30(M, C, V)
    TGT10 = bh.build_target(C, O, 10).values

    out = [f"# EGX Improve - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- احتفاظ 10 أيام | تكلفة {a.cost_bps:.1f} bps في الاتجاه | EGX30 من: "
           f"{'Yahoo' if egx_src == 'yahoo' else 'مؤشر تقريبي'} | كل الموديلات بوزن الأحدث (half-life 365)",
           f"- الأرقام: {RECENT}+ ، bps للصفقة بعد التكلفة، أعلى فئة من الأسهم مقابل متوسط السوق", ""]
    head = ["| الإعداد | أيام | قبل التكلفة | بعد التكلفة | t | أيام كسبانة |", "|---|---|---|---|---|---|"]

    # ---- 1) حد السيولة
    res1, best = [], None
    keep = {}
    for tov in a.turnovers:
        print(f"[improve] min_turnover={tov:,}")
        Xcs, feat_ok = setup(C, H, L, V, O, tov, M)
        pred, t0 = fit_pred(Xcs, feat_ok, C, O, 10, a)
        E = feat_ok & np.isfinite(TGT10)
        r = eval_pred(pred, TGT10, E, t0, T, dates, cost)
        res1.append((tov, row(f"حد سيولة {tov / 1e6:g} مليون", r), r))
        keep[tov] = (Xcs, feat_ok, pred, t0, E)
    out += ["## 1) حد السيولة", "", *head, *[x[1] for x in res1], ""]
    scored = [(x[2]["net"], x[0]) for x in res1 if x[2]]
    best_tov = max(scored)[1] if scored else a.turnovers[0]
    Xcs, feat_ok, pred, t0, E = keep[best_tov]
    out += [f"> الأفضل: حد سيولة {best_tov / 1e6:g} مليون. باقي التجارب عليه.", ""]

    # ---- 2) حجم الفئة المختارة
    out += ["## 2) حجم الفئة المختارة", "", *head]
    for fr in (0.1, 0.2, 0.3):
        out.append(row(f"أعلى {fr * 100:.0f}%", eval_pred(pred, TGT10, E, t0, T, dates, cost, frac=fr)))
    out.append("")

    # ---- 3) فلتر السوق
    out += ["## 3) فلتر السوق (متدخلش لو وسيط آخر 5 أيام نازل)", "", *head]
    out.append(row("من غير فلتر", eval_pred(pred, TGT10, E, t0, T, dates, cost)))
    for thr in (-0.02, -0.04):
        skip = mk5 <= thr
        r = eval_pred(pred, TGT10, E, t0, T, dates, cost, skip=skip)
        out.append(row(f"تجنب لو آخر 5 أيام ≤ {thr * 100:.0f}%", r))
    out.append("")

    # ---- 4) ensemble فترات
    print("[improve] ensemble 5/10/20")
    preds = []
    for h in (5, 10, 20):
        p, t0h = fit_pred(Xcs, feat_ok, C, O, h, a)
        preds.append(p)
    ens = rank_avg(preds)
    t0e = max(t0, t0h)
    out += ["## 4) ensemble (متوسط ترتيب موديلات 5 و10 و20 يوم) على هدف 10 أيام", "", *head,
            row("موديل 10 أيام لوحده", eval_pred(pred, TGT10, E, t0e, T, dates, cost)),
            row("موديل 5 أيام", eval_pred(preds[0], TGT10, E, t0e, T, dates, cost)),
            row("موديل 20 يوم", eval_pred(preds[2], TGT10, E, t0e, T, dates, cost)),
            row("ensemble", eval_pred(ens, TGT10, E, t0e, T, dates, cost)), ""]

    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "improve_latest" if a.source == "yfinance" else f"improve_{a.source}"
    with open(os.path.join(bt.RES_DIR, f"report_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--turnovers", type=lambda s: [int(float(x)) for x in s.split(",") if x],
                    default=[1_000_000, 3_000_000, 6_000_000])
    a = ap.parse_args()
    try:
        run(a)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        with open(os.path.join(bt.RES_DIR, "error_improve_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M}\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
