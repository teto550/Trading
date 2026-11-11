#!/usr/bin/env python3
"""وضع الثقة العالية: السهم يتختار بس لو الموديلات (XGB+LightGBM+CatBoost) وفترات 20/40/60 كلهم متفقين.
بيقيس نسبة الصح الحقيقية (صفقات 2022+ بعد تكلفتك) لكل مستوى اتفاق."""
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
KINDS = ("xgb", "lgbm", "cat")
HS = (20, 40, 60)


def wilson(k, n, z=1.96):
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - w, c + w


def evaluate(sel, TGT, E, t0, dates, h, cost):
    """sel: (T,N) bool لاختيارات. بيرجّع إحصائيات الصفقات (كل يوم بداية) + فاصل ثقة بعدد مستقل ≈ n/h."""
    ex, ab, days = [], [], 0
    for t in range(t0, len(dates) - 1):
        if dates[t].year < RECENT:
            continue
        m = E[t]
        if m.sum() < 30:
            continue
        s = sel[t] & m
        if not s.any():
            continue
        mk = TGT[t, m].mean()
        r = TGT[t, s]
        ex += list(r - mk - 2 * cost)
        ab += list(r - 2 * cost)
        days += 1
    n = len(ex)
    if n < 30:
        return None
    ex, ab = np.array(ex), np.array(ab)
    neff = max(int(n / h), 1)
    k_ex, k_ab = int((ex > 0).sum()), int((ab > 0).sum())
    lo, hi = wilson(round((k_ex / n) * neff), neff)
    return {"n": n, "days": days, "ex_net": ex.mean() * 1e4, "p_ex": k_ex / n * 100, "p_abs": k_ab / n * 100,
            "ci": (lo * 100, hi * 100), "neff": neff}


def row(label, r):
    if not r:
        return f"| {label} | - | - | - | - | - | - |"
    return (f"| {label} | {r['n']} ({r['days']} يوم) | {r['ex_net']:+.0f} | {r['p_ex']:.0f}% | "
            f"{r['ci'][0]:.0f}–{r['ci'][1]:.0f}% | {r['p_abs']:.0f}% | ~{r['neff']} |")


def run(a):
    cost = a.cost_bps / 1e4
    panel = bt.fetch_yfinance(bt.load_tickers()) if a.source == "yfinance" else bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    dates, (T, N) = C.index, C.shape
    f0, mkt0, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, 1_000_000)
    M = macro.fetch_macro(fresh=True) if a.source == "yfinance" else macro.synthetic_macro(mkt0["mkt1"])
    M, _src = macro.ensure_egx30(M, C, V)
    X, _ok = im.setup(C, H, L, V, O, 1_000_000, M)
    XA, _names = al.build_alpha(C, H, L, V, O, liquid, bt.csrank)
    X2 = np.concatenate([X, XA], axis=2)
    ok2 = np.isfinite(X2).all(axis=2) & liquid.values & ~lock.values
    R, tg, t0s = {}, {}, []
    for h in HS:
        TGT, E, fv, Y = i2.prep(C, O, h, ok2)
        tg[h] = (TGT, E)
        for k in KINDS:
            print(f"[conviction] h={h} {k}")
            p, t0 = i2.walk(k, X2, Y, E, fv, a, h - 1)
            R[(h, k)] = pd.DataFrame(p).rank(axis=1, pct=True).values
            t0s.append(t0)
    t0 = max(t0s)
    out = [f"# EGX Conviction Test - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- XGB+LightGBM+CatBoost × فترات 20/40/60، Alpha158، تكلفة {a.cost_bps:.1f} bps في الاتجاه، صفقات 2022+",
           "- 'صح' = السهم غلب متوسط السوق بعد التكلفة | 'ربح مطلق' = عائد السهم بعد التكلفة موجب | الفاصل = 95% بعدد صفقات مستقلة تقريبي (الصفقات متداخلة)", ""]
    head = ["| الإعداد | صفقات | زيادة (bps) | صح (غلب السوق) | فاصل الثقة | ربح مطلق | صفقات مستقلة |", "|---|---|---|---|---|---|---|"]
    for h in HS:
        TGT, E = tg[h]
        ens = np.nanmean(np.stack([R[(h, k)] for k in KINDS]), axis=0)
        allr = np.stack([R[(hh, k)] for hh in HS for k in KINDS])      # 9 ترتيب
        out += [f"## هدف {h} يوم", "", *head]
        out.append(row("أعلى 20% (ensemble الموديلات)", evaluate(ens >= 0.8, TGT, E, t0, dates, h, cost)))
        out.append(row("أعلى 10% (ensemble الموديلات)", evaluate(ens >= 0.9, TGT, E, t0, dates, h, cost)))
        for q, lab in ((0.8, "20%"), (0.9, "10%")):
            agree3 = np.all(np.stack([R[(h, k)] for k in KINDS]) >= q, axis=0)
            out.append(row(f"3 موديلات متفقين - أعلى {lab}", evaluate(agree3, TGT, E, t0, dates, h, cost)))
        for q, lab in ((0.8, "20%"), (0.9, "10%")):
            agree9 = np.all(allr >= q, axis=0)
            out.append(row(f"كل الـ9 (3 موديلات × 3 فترات) متفقين - أعلى {lab}", evaluate(agree9, TGT, E, t0, dates, h, cost)))
        out.append("")
        out += ["الأسهم اللي الـ9 متفقين إنها الأضعف (أسفل 20%):", ""]
        agree9d = np.all(allr <= 0.2, axis=0)
        r = evaluate(agree9d, TGT, E, t0, dates, h, cost)
        if r:
            # للتحويل: الصح هنا = السهم خسر قدام السوق (قبل تكلفة بيع، مش بنحسب تكلفة)
            ex_raw = -(r["ex_net"] / 1e4 + 2 * cost)
            out.append(f"- {r['n']} صفقة ({r['days']} يوم): متوسط أداء السهم عن السوق {-ex_raw * 1e4:+.0f} bps (بدون تكلفة).")
        out.append("")
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "conviction_latest" if a.source == "yfinance" else f"conviction_{a.source}"
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
        open(os.path.join(bt.RES_DIR, "error_conviction_latest.txt"), "w", encoding="utf-8").write(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
