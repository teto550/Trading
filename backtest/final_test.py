#!/usr/bin/env python3
"""تحقق مستقل للفترات الطويلة: صفقات مش متداخلة (كل h يوم) + أعلى/أقل سوق + ensemble فترات 20/40/60."""
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


def cohorts(pred, TGT, E, t0, dates, h, frac, off):
    """صفقات مش متداخلة: يوم كل h يوم بادئ من t0+off. بيرجّع قايمة (تاريخ, عائد الاختيارات, عائد السوق)."""
    out = []
    for t in range(t0 + off, len(dates) - 1, h):
        if dates[t].year < RECENT:
            continue
        m = E[t] & np.isfinite(pred[t])
        n = int(m.sum())
        if n < 30:
            continue
        cols = np.where(m)[0]
        s, y = pred[t, cols], TGT[t, cols]
        top = cols[np.argsort(-s)[: max(int(n * frac), 5)]]
        out.append((t, float(TGT[t, top].mean()), float(y.mean())))
    return out


def summarize(pred, TGT, E, t0, dates, h, frac, cost):
    per_off = []
    allrows = []
    for off in range(h):
        c = cohorts(pred, TGT, E, t0, dates, h, frac, off)
        if len(c) < 8:
            continue
        ex = np.array([a - b for _, a, b in c]) - 2 * cost
        per_off.append((ex.mean(), ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)) + 1e-12), len(ex)))
        allrows += [(a - 2 * cost, b, a - b - 2 * cost) for _, a, b in c]
    if not per_off:
        return None
    A = np.array(allrows)
    up, dn = A[A[:, 1] > 0], A[A[:, 1] <= 0]
    return {"n_indep": int(np.mean([p[2] for p in per_off])), "ex_net": np.mean([p[0] for p in per_off]) * 1e4,
            "t": np.mean([p[1] for p in per_off]), "abs_net": A[:, 0].mean() * 100, "mkt": A[:, 1].mean() * 100,
            "win": float((A[:, 2] > 0).mean() * 100),
            "ex_up": float(up[:, 2].mean() * 1e4) if len(up) else float("nan"), "n_up": len(up),
            "ex_dn": float(dn[:, 2].mean() * 1e4) if len(dn) else float("nan"), "n_dn": len(dn),
            "worst": float(A[:, 2].min() * 100)}


def row(label, r):
    if not r:
        return f"| {label} | - | - | - | - | - | - | - | - |"
    return (f"| {label} | {r['n_indep']} | {r['ex_net']:+.0f} | {r['t']:+.1f} | {r['win']:.0f}% | {r['abs_net']:+.1f}% "
            f"(سوق {r['mkt']:+.1f}%) | {r['ex_up']:+.0f} ({r['n_up']}) | {r['ex_dn']:+.0f} ({r['n_dn']}) | {r['worst']:+.1f}% |")


def rank_avg(preds):
    st = np.stack([pd.DataFrame(p).rank(axis=1, pct=True).values for p in preds])
    out = np.full_like(preds[0], np.nan)
    ok = np.isfinite(st).all(axis=0)
    out[ok] = st.mean(axis=0)[ok]
    return out


def run(a):
    cost = a.cost_bps / 1e4
    panel = bt.fetch_yfinance(bt.load_tickers()) if a.source == "yfinance" else bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    dates, (T, N) = C.index, C.shape
    f0, mkt0, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, 1_000_000)
    M = macro.fetch_macro(fresh=True) if a.source == "yfinance" else macro.synthetic_macro(mkt0["mkt1"])
    M, _src = macro.ensure_egx30(M, C, V)
    X, _ok = im.setup(C, H, L, V, O, 1_000_000, M)
    XA, names = al.build_alpha(C, H, L, V, O, liquid, bt.csrank)
    X2 = np.concatenate([X, XA], axis=2)
    ok2 = np.isfinite(X2).all(axis=2) & liquid.values & ~lock.values
    preds, tg = {}, {}
    for h in (20, 40, 60):
        print(f"[final] h={h}")
        TGT, E, fv, Y = i2.prep(C, O, h, ok2)
        p, t0 = i2.walk("lgbm", X2, Y, E, fv, a, h - 1)
        preds[h], tg[h] = p, (TGT, E, t0)
    out = [f"# EGX Final Test - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- LightGBM + Alpha158، تكلفة {a.cost_bps:.1f} bps في الاتجاه، 2022+",
           "- صفقات **مش متداخلة** (كل h يوم)، والأرقام متوسط على كل نقاط البداية الممكنة. 'زيادة' = صافي bps بعد التكلفة عن متوسط السوق للصفقة",
           "- 'سوق طالع/نازل' = متوسط الزيادة لما السوق كسب/خسر في نفس الفترة (بين القوسين عدد الصفقات)", ""]
    head = ["| الإعداد | صفقات مستقلة | زيادة (bps) | t | صفقات غلبت السوق | عائد مطلق صافي | زيادة لما السوق طالع | زيادة لما السوق نازل | أسوأ صفقة (زيادة) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for h in (20, 40, 60):
        TGT, E, t0 = tg[h]
        out += [f"## احتفاظ {h} يوم", "", *head]
        for fr in (0.1, 0.2):
            out.append(row(f"أعلى {fr * 100:.0f}%", summarize(preds[h], TGT, E, t0, dates, h, fr, cost)))
        out.append("")
    out += ["## ensemble ترتيب موديلات 20+40+60 (على هدف 40 و60)", "", *head]
    ens = rank_avg([preds[20], preds[40], preds[60]])
    for h in (40, 60):
        TGT, E, t0 = tg[h]
        t0e = max(tg[20][2], tg[40][2], tg[60][2])
        for fr in (0.1, 0.2):
            out.append(row(f"ensemble على {h} يوم - أعلى {fr * 100:.0f}%", summarize(ens, TGT, E, t0e, dates, h, fr, cost)))
    out.append("")
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "final_latest" if a.source == "yfinance" else f"final_{a.source}"
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
        open(os.path.join(bt.RES_DIR, "error_final_latest.txt"), "w", encoding="utf-8").write(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
