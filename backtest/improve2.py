#!/usr/bin/env python3
"""
تجارب تحسين الجولة التانية (LightGBM، بوزن الأحدث، تكلفة 20 bps، 2022+، bps للصفقة بعد التكلفة):
  A) الأساسي
  B) + مدخلات Alpha158 (حوالي 55 مدخل إضافي)
  C) bagging: متوسط 5 seeds
  D) فترات أطول: 10 / 20 / 40 / 60 يوم (التكلفة بتتوزع على فترة أطول)
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
import alpha as al  # noqa: E402
import backtest as bt  # noqa: E402
import backtest_horizons as bh  # noqa: E402
import improve as im  # noqa: E402
import macro  # noqa: E402
import models  # noqa: E402


def walk(kind, X, Y, E, first_valid, a, embargo, seed=0):
    T, N, K = X.shape
    pred = np.full((T, N), np.nan)
    t0 = first_valid + a.min_train
    for b in range(t0, T - 1, a.block):
        cut = b - embargo
        tr = E[:cut]
        if tr.sum() < 5000:
            continue
        age = (cut - 1 - np.arange(cut)).astype(float)
        w = np.repeat((0.5 ** (age / 365.0))[:, None], N, axis=1)[tr]
        m = models.make(kind, a.trees, seed)
        m.fit(X[:cut][tr], Y[:cut][tr], sample_weight=w)
        for t in range(b, min(b + a.block, T - 1)):
            cols = np.where(np.isfinite(X[t]).all(axis=1))[0]
            if len(cols):
                pred[t, cols] = m.predict(X[t, cols])
    return pred, t0


def prep(C, O, h, feat_ok):
    TGT = bh.build_target(C, O, h).values
    E = feat_ok & np.isfinite(TGT)
    elig = E.sum(axis=1)
    mc = int(np.clip(0.5 * np.percentile(elig[elig > 0], 75), 20, bt.MIN_CROSS))
    fv = int(np.where(elig >= mc)[0][0])
    Y = np.nan_to_num((pd.DataFrame(np.where(E, TGT, np.nan)).rank(axis=1, pct=True) - 0.5).values.astype(np.float32))
    return TGT, E, fv, Y


def run(a):
    cost = a.cost_bps / 1e4
    panel = bt.fetch_yfinance(bt.load_tickers()) if a.source == "yfinance" else bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    dates, (T, N) = C.index, C.shape
    f0, mkt0, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, 1_000_000)
    M = macro.fetch_macro(fresh=True) if a.source == "yfinance" else macro.synthetic_macro(mkt0["mkt1"])
    M, _src = macro.ensure_egx30(M, C, V)
    X, feat_ok = im.setup(C, H, L, V, O, 1_000_000, M)
    print(f"[improve2] مدخلات أساسية {X.shape[2]}")
    XA, names = al.build_alpha(C, H, L, V, O, liquid, bt.csrank)
    X2 = np.concatenate([X, XA], axis=2)
    ok2 = np.isfinite(X2).all(axis=2) & liquid.values & ~lock.values
    print(f"[improve2] + {len(names)} مدخل Alpha158 = {X2.shape[2]}")
    out = [f"# EGX Improve2 - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- LightGBM بوزن الأحدث | تكلفة {a.cost_bps:.1f} bps | 2022+ | bps للصفقة بعد التكلفة", ""]
    head = ["| الإعداد | أيام | قبل التكلفة | بعد التكلفة | t | أيام كسبانة |", "|---|---|---|---|---|---|"]

    def ev(pred, TGT, E, t0, tag):
        res = []
        for fr in (0.1, 0.2):
            res.append(im.row(f"{tag} - أعلى {fr * 100:.0f}%", im.eval_pred(pred, TGT, E, t0, T, dates, cost, frac=fr)))
        return res

    h = 20
    TGT, E, fv, Y = prep(C, O, h, feat_ok)
    print("[improve2] A")
    pA, t0 = walk("lgbm", X, Y, E, fv, a, h - 1)
    TGT2, E2, fv2, Y2 = prep(C, O, h, ok2)
    print("[improve2] B")
    pB, t0b = walk("lgbm", X2, Y2, E2, fv2, a, h - 1)
    print("[improve2] C (bagging)")
    ps = [walk("lgbm", X, Y, E, fv, a, h - 1, seed=s)[0] for s in range(1, 5)] + [pA]
    pC = np.nanmean(np.stack(ps), axis=0)
    out += [f"## احتفاظ {h} يوم", "", *head, *ev(pA, TGT, E, t0, "A الأساسي"),
            *ev(pB, TGT2, E2, max(t0, t0b), f"B + Alpha158 ({len(names)} مدخل)"),
            *ev(pC, TGT, E, t0, "C bagging 5 seeds"), ""]

    out += ["## فترات احتفاظ أطول (الأساسي، LightGBM)", "", *head]
    for hh in (10, 20, 40, 60):
        if hh == 20:
            pp, TG, EE, tt = pA, TGT, E, t0
        else:
            TG, EE, fvh, Yh = prep(C, O, hh, feat_ok)
            print(f"[improve2] D h={hh}")
            pp, tt = walk("lgbm", X, Yh, EE, fvh, a, hh - 1)
        out += ev(pp, TG, EE, tt, f"{hh} يوم")
    out.append("")
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "improve2_latest" if a.source == "yfinance" else f"improve2_{a.source}"
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
        open(os.path.join(bt.RES_DIR, "error_improve2_latest.txt"), "w", encoding="utf-8").write(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
