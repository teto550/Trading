#!/usr/bin/env python3
"""فلتر اتجاه السوق: هل التداول بس لما مؤشر السوق فوق متوسطه (100/200 يوم) بيحسّن النتيجة؟ صفقات مش متداخلة 2022+، بعد التكلفة."""
import argparse
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import alpha as al  # noqa: E402
import backtest as bt  # noqa: E402
import backtest_horizons as bh  # noqa: E402
import improve as im  # noqa: E402
import improve2 as i2  # noqa: E402
import macro  # noqa: E402

RECENT = 2022


def cohorts(sc, TGT, E, t0, dates, h, frac, off):
    out = []
    for t in range(t0 + off, len(dates) - 1, h):
        if dates[t].year < RECENT:
            continue
        m = E[t] & np.isfinite(sc[t])
        n = int(m.sum())
        if n < 30:
            continue
        cols = np.where(m)[0]
        top = cols[np.argsort(-sc[t, cols])[: max(int(n * frac), 5)]]
        out.append((t, float(TGT[t, top].mean()), float(TGT[t, cols].mean())))
    return out


def stats(rows, cost, flt=None):
    """rows: (t, pick, mkt). flt: bool لكل t -> داخل السوق؛ برا = كاش 0."""
    if not rows:
        return None
    pick = np.array([r[1] for r in rows]) - 2 * cost
    mk = np.array([r[2] for r in rows])
    on = np.array([bool(flt[r[0]]) if flt is not None else True for r in rows])
    ret = np.where(on, pick, 0.0)
    return {"n": len(rows), "n_on": int(on.sum()), "avg": ret.mean() * 100, "mkt": mk.mean() * 100,
            "ex_on": float((pick - mk)[on].mean() * 1e4) if on.any() else float("nan"),
            "win_on": float((pick[on] > mk[on]).mean() * 100) if on.any() else float("nan"),
            "worst": float(ret.min() * 100), "sd": float(ret.std() * 100)}


def row(label, r):
    if not r:
        return f"| {label} | - | - | - | - | - | - |"
    return (f"| {label} | {r['n_on']}/{r['n']} | {r['avg']:+.2f}% | {r['mkt']:+.2f}% | {r['ex_on']:+.0f} | "
            f"{r['win_on']:.0f}% | {r['worst']:+.1f}% |")


def run(a):
    cost = a.cost_bps / 1e4
    panel = bt.fetch_yfinance(bt.load_tickers()) if a.source == "yfinance" else bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    dates = C.index
    f0, mkt0, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, 1_000_000)
    M = macro.fetch_macro(fresh=True) if a.source == "yfinance" else macro.synthetic_macro(mkt0["mkt1"])
    M, _src = macro.ensure_egx30(M, C, V)
    X, _ok = im.setup(C, H, L, V, O, 1_000_000, M)
    XA, _names = al.build_alpha(C, H, L, V, O, liquid, bt.csrank)
    X2 = np.concatenate([X, XA], axis=2)
    ok2 = np.isfinite(X2).all(axis=2) & liquid.values & ~lock.values
    idx = (1 + pd.Series(mkt0["mkt1"].values).fillna(0).clip(-0.15, 0.15)).cumprod()
    f200 = (idx > idx.rolling(200, min_periods=150).mean()).values
    f100 = (idx > idx.rolling(100, min_periods=80).mean()).values
    mom60 = (idx / idx.shift(60) - 1).values
    f60 = np.nan_to_num(mom60) > 0
    P, tg, t0s = {}, {}, []
    for h in (20, 40, 60):
        print(f"[trend] h={h}")
        TGT, E, fv, Y = i2.prep(C, O, h, ok2)
        p, t0 = i2.walk("lgbm", X2, Y, E, fv, a, h - 1)
        P[h] = pd.DataFrame(p).rank(axis=1, pct=True).values
        tg[h] = (TGT, E)
        t0s.append(t0)
    t0 = max(t0s)
    ens = np.nanmean(np.stack([P[20], P[40], P[60]]), axis=0)
    out = [f"# EGX Trend Filter Test - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- الموديل: ensemble 20/40/60 (LightGBM+Alpha158)، أعلى 20%، تكلفة {a.cost_bps:.1f} bps في الاتجاه، 2022+، صفقات مش متداخلة",
           "- الفلتر: مؤشر متوسط الأسهم (equal-weight) فوق متوسطه المتحرك وقت الدخول. 'برا' = كاش (عائد 0)",
           "- 'عائد' = متوسط عائد الصفقة بعد التكلفة (كل الفترات بما فيها الكاش) | 'سوق' = لو اشتريت متوسط الأسهم دايمًا | 'زيادة' = bps عن السوق وقت ما الفلتر دخل بس | 'أسوأ' = أسوأ فترة", ""]
    head = ["| الإعداد | صفقات داخل/كل | عائد | سوق | زيادة (bps) | غلب السوق | أسوأ فترة |", "|---|---|---|---|---|---|---|"]
    flts = [("بدون فلتر", None), ("السوق فوق متوسط 200 يوم", f200), ("السوق فوق متوسط 100 يوم", f100), ("السوق صاعد آخر 60 يوم", f60)]
    for h in (20, 40, 60):
        TGT, E = tg[h]
        rows = []
        for off in range(h):
            rows += cohorts(ens, TGT, E, t0, dates, h, 0.2, off)
        out += [f"## احتفاظ {h} يوم", "", *head]
        for lab, flt in flts:
            out.append(row(lab, stats(rows, cost, flt)))
        # عكسي: لما السوق تحت المتوسط (فين الميزة؟)
        out.append(row("(مرجع) بس لما السوق تحت متوسط 200", stats(rows, cost, ~f200)))
        out.append("")
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "trend_latest" if a.source == "yfinance" else f"trend_{a.source}"
    open(os.path.join(bt.RES_DIR, f"report_{tag}.md"), "w", encoding="utf-8").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--trees", type=int, default=150)
    a = ap.parse_args()
    try:
        run(a)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        open(os.path.join(bt.RES_DIR, "error_trend_latest.txt"), "w", encoding="utf-8").write(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
