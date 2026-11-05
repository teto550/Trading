#!/usr/bin/env python3
"""
تجربة على طريقة أفضل نتايج معيار Qlib (DoubleEnsemble = LightGBM + إعادة وزن العينات + تجميع):
XGBoost الحالي مقابل LightGBM و CatBoost ومتوسط ترتيبهم (ensemble)، بنفس وزن الأحدث، على احتفاظ 10 و20 يوم،
بتكلفة 20 bps في الاتجاه، 2022+.
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
import macro  # noqa: E402
import improve as im  # noqa: E402

RECENT = 2022


def make(kind, trees, seed=0):
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=trees, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
                                min_child_weight=50, reg_lambda=10.0, tree_method="hist", n_jobs=2, random_state=seed)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=trees, num_leaves=15, learning_rate=0.05, subsample=0.7, subsample_freq=1,
                                 colsample_bytree=0.7, min_child_samples=100, reg_lambda=10.0, n_jobs=2,
                                 random_state=seed, verbose=-1)
    import catboost as cb
    return cb.CatBoostRegressor(iterations=trees, depth=4, learning_rate=0.06, l2_leaf_reg=10, thread_count=2,
                                random_seed=seed, verbose=0, allow_writing_files=False)


def walk(kind, Xcs, Y, E, first_valid, a, embargo):
    T, N, K = Xcs.shape
    pred = np.full((T, N), np.nan)
    t0 = first_valid + a.min_train
    for b in range(t0, T - 1, a.block):
        cut = b - embargo
        tr = E[:cut]
        if tr.sum() < 5000:
            continue
        age = (cut - 1 - np.arange(cut)).astype(float)
        w = np.repeat((0.5 ** (age / 365.0))[:, None], N, axis=1)[tr]
        m = make(kind, a.trees)
        m.fit(Xcs[:cut][tr], Y[:cut][tr], sample_weight=w)
        for t in range(b, min(b + a.block, T - 1)):
            cols = np.where(np.isfinite(Xcs[t]).all(axis=1))[0]
            if len(cols):
                pred[t, cols] = m.predict(Xcs[t, cols])
    return pred, t0


def run(a):
    cost = a.cost_bps / 1e4
    panel = bt.fetch_yfinance(bt.load_tickers()) if a.source == "yfinance" else bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    dates, (T, N) = C.index, C.shape
    f0, mkt0, *_ = bt.build_features(C, H, L, V, O, 1_000_000)
    M = macro.fetch_macro(fresh=True) if a.source == "yfinance" else macro.synthetic_macro(mkt0["mkt1"])
    M, egx_src = macro.ensure_egx30(M, C, V)
    Xcs, feat_ok = im.setup(C, H, L, V, O, 1_000_000, M)
    out = [f"# EGX Ensemble Test - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- تكلفة {a.cost_bps:.1f} bps في الاتجاه | {RECENT}+ | bps للصفقة بعد التكلفة (أعلى 20% مقابل متوسط السوق) | كلهم بوزن الأحدث", ""]
    for h in a.horizons:
        print(f"[ensemble] h={h}")
        TGT = bh.build_target(C, O, h).values
        E = feat_ok & np.isfinite(TGT)
        elig = E.sum(axis=1)
        mc = int(np.clip(0.5 * np.percentile(elig[elig > 0], 75), 20, bt.MIN_CROSS))
        fv = int(np.where(elig >= mc)[0][0])
        Y = np.nan_to_num((pd.DataFrame(np.where(E, TGT, np.nan)).rank(axis=1, pct=True) - 0.5).values.astype(np.float32))
        preds, t0 = {}, None
        for k in ("xgb", "lgbm", "cat"):
            try:
                preds[k], t0 = walk(k, Xcs, Y, E, fv, a, h - 1)
            except Exception as e:  # noqa: BLE001
                print(f"[ensemble] {k} فشل: {e}")
        ens = im.rank_avg(list(preds.values()))
        rows = [(k, p) for k, p in preds.items()] + [("ensemble (متوسط الترتيب)", ens)]
        out += [f"## احتفاظ {h} يوم", "", "| الموديل | أيام | قبل التكلفة | بعد التكلفة | t | أيام كسبانة |", "|---|---|---|---|---|---|"]
        for name, p in rows:
            for fr in (0.2, 0.1):
                r = im.eval_pred(p, TGT, E, t0, T, dates, cost, frac=fr)
                tag = f"{name} - أعلى {fr * 100:.0f}%"
                out.append(im.row(tag, r))
        out.append("")
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "ensemble_latest" if a.source == "yfinance" else f"ensemble_{a.source}"
    open(os.path.join(bt.RES_DIR, f"report_{tag}.md"), "w", encoding="utf-8").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--horizons", type=lambda s: [int(x) for x in s.split(",") if x], default=[10, 20])
    a = ap.parse_args()
    try:
        run(a)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        open(os.path.join(bt.RES_DIR, "error_ensemble_latest.txt"), "w", encoding="utf-8").write(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
