#!/usr/bin/env python3
"""مقارنة موديلات تصنيف شراء/بيع (احتمال إن السهم يقفل بكره فوق الافتتاح) على فترتين خارج التدريب."""
import argparse
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import alpha  # noqa: E402
import backtest as bt  # noqa: E402
import backtest_horizons as bh  # noqa: E402
import direction as dr  # noqa: E402


def mk(kind, trees):
    if kind == "xgb":
        return dr._model(trees)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=trees, max_depth=3, num_leaves=8, learning_rate=0.05,
                                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, min_child_samples=50,
                                  reg_lambda=10.0, n_jobs=2, random_state=0, verbose=-1)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=trees, depth=4, learning_rate=0.06, l2_leaf_reg=10, thread_count=2,
                              random_seed=0, verbose=0, allow_writing_files=False)


def auc(y, p):
    r = pd.Series(p).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(X_base, X_all, ret, E, dates_ok, start, end, trees, day_mean):
    T = X_base.shape[0]
    cut = start - 1
    Etr = E.copy()
    Etr[cut:] = False
    Ete = np.zeros_like(E)
    Ete[start:end] = E[start:end]
    ytr = (ret[Etr] > 0).astype(int)
    y = (ret[Ete] > 0).astype(int)
    ex = (ret - day_mean[:, None])[Ete]          # عائد السهم - متوسط السوق نفس اليوم
    w = dr.recency_w(Etr)
    probs = {}
    for name, X, kind in (("XGB (الحالي)", X_base, "xgb"), ("XGB+Alpha", X_all, "xgb"),
                          ("LightGBM+Alpha", X_all, "lgbm"), ("CatBoost+Alpha", X_all, "cat")):
        m = mk(kind, trees)
        m.fit(X[Etr], ytr, sample_weight=w)
        probs[name] = m.predict_proba(X[Ete])[:, 1]
        print(f"  [{name}] تم")
    probs["متوسط الـ3 (+Alpha)"] = np.mean([probs["XGB+Alpha"], probs["LightGBM+Alpha"], probs["CatBoost+Alpha"]], axis=0)
    up = float(y.mean())
    rows = []
    for name, p in probs.items():
        # عتبة نسبية (أعلى/أدنى 15% احتمال) عشان نقارن الموديلات حتى لو معايرتها مختلفة
        hi, lo = np.quantile(p, 0.85), np.quantile(p, 0.15)
        b, s = p >= hi, p <= lo
        rows.append({
            "name": name, "auc": auc(y, p), "acc": float(((p >= 0.5) == y).mean() * 100),
            "buy_hit": float(y[b].mean() * 100), "buy_ex": float(ex[b].mean() * 1e4),
            "sell_hit": float((1 - y[s]).mean() * 100), "sell_ex": float(-ex[s].mean() * 1e4),
            "buy55": float(y[p >= 0.55].mean() * 100) if (p >= 0.55).sum() > 30 else float("nan"),
            "n55": int((p >= 0.55).sum()),
            "sell45": float((1 - y[p <= 0.45]).mean() * 100) if (p <= 0.45).sum() > 30 else float("nan"),
            "n45": int((p <= 0.45).sum())})
    return rows, len(y), max(up, 1 - up) * 100, up * 100


def table(rows):
    L = ["| الموديل | AUC | أعلى 15%: صح% | أعلى 15%: زيادة عن السوق (bps) | أدنى 15%: صح% | أدنى 15%: زيادة (bps) | شراء ≥0.55 (عدد) | بيع ≤0.45 (عدد) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['auc']:.3f} | {r['buy_hit']:.1f} | {r['buy_ex']:+.0f} | {r['sell_hit']:.1f} | "
                 f"{r['sell_ex']:+.0f} | {r['buy55']:.1f} ({r['n55']}) | {r['sell45']:.1f} ({r['n45']}) |")
    return L


def run(a):
    panel = bt.fetch_yfinance(bt.load_tickers()) if a.source == "yfinance" else bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    f, mkt, _n, nxt_oc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, 1_000_000)
    T, N = C.shape
    cs = {k: bt.csrank(f[k], liquid).values for k in bt.XF}
    Xb = np.stack([cs[k] for k in bt.XF] + [np.repeat(mkt[k].values[:, None], N, axis=1) for k in bt.MF], axis=2).astype(np.float32)
    XA, _an = alpha.build_alpha(C, H, L, V, O, liquid, bt.csrank)
    Xa = np.concatenate([Xb, XA], axis=2)
    ok = np.isfinite(Xa).all(axis=2) & liquid.values & ~lock.values
    ret = nxt_oc.values
    E = ok & np.isfinite(ret)
    day_mean = np.where(E.sum(1) > 0, np.nanmean(np.where(E, ret, np.nan), axis=1), 0.0)
    out = [f"# EGX Direction Test - {datetime.now():%Y-%m-%d %H:%M}", "",
           "- الهدف: السهم يقفل بكره فوق افتتاحه؟ (نفس اللي الموديل الحالي بيتوقعه) | التدريب قبل الفترة بيوم | وزن الأحدث 365 يوم",
           "- 'زيادة عن السوق' = عائد السهم - متوسط الأسهم نفس اليوم، قبل أي تكلفة (تكلفتك ~40 bps للصفقة)", ""]
    for label, (s, e) in (("آخر 250 يوم", (T - 250, T)), ("250 يوم قبلها", (T - 500, T - 250))):
        print(f"[dir_test] {label}")
        rows, n, base, up = evaluate(Xb, Xa, ret, E, None, s, e, a.trees, day_mean)
        out += [f"## {label}", f"- {n} مشاهدة | نسبة الطالع {up:.1f}% | أحسن تخمين ثابت {base:.1f}%", "", *table(rows), ""]
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "dir_latest" if a.source == "yfinance" else f"dir_{a.source}"
    with open(os.path.join(bt.RES_DIR, f"report_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--trees", type=int, default=150)
    a = ap.parse_args()
    try:
        run(a)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        with open(os.path.join(bt.RES_DIR, "error_dir_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M}\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
