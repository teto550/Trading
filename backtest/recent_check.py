#!/usr/bin/env python3
"""
فحص تطابق: الموديل لو اتدرّب على اللي قبل 23 أكتوبر بس، كان هيعمل إيه في الجلسات اللي عدّت؟
لو النتيجة هنا سالبة زي الاختبار الحي، يبقى الفترة نفسها (صدمة السوق) هي السبب.
لو موجبة، يبقى فيه فرق بين الباكتست والتشغيل الحي (بيانات آخر يوم، التنفيذ...) ولازم نصلحه.
"""
import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as bt  # noqa: E402
import backtest_horizons as bh  # noqa: E402
import macro  # noqa: E402


def run(a):
    import xgboost as xgb
    panel = bt.fetch_yfinance(bt.load_tickers())
    C, H, L, V, O = bt.clean_panel(panel)
    dates, (T, N) = C.index, C.shape
    f, mkt, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, 1_000_000)
    lr = np.log1p(f["r1"])
    f["r60"] = np.expm1(lr.rolling(60, min_periods=60).sum())
    f["r120"] = np.expm1(lr.rolling(120, min_periods=120).sum())
    M, _src = macro.ensure_egx30(macro.fetch_macro(fresh=True), C, V)
    Xcs, names = macro.build_X(f, mkt, liquid, N, bh.XF2, bt.MF, M, bt.csrank)
    feat_ok = np.isfinite(Xcs).all(axis=2) & liquid.values & ~lock.values
    t_start = int(np.searchsorted(dates, pd.Timestamp(a.start)))
    print(f"[recent] آخر يوم بيانات {dates[-1].date()} | بداية الفحص {dates[t_start].date()} ({T - t_start} يوم)")
    cc = (C.shift(-1) / C - 1).values                      # قفلة لقفلة (زي التقييم الحي)
    out = [f"# فحص الجلسات الأخيرة (موديل متدرب على قبل {dates[t_start].date()} بس)", "",
           "excess = متوسط أعلى 10 أسهم - متوسط السوق (٪)، oc = شراء الافتتاح وبيع القفلة، cc = قفلة لقفلة", ""]
    for h in a.horizons:
        TGT = bh.build_target(C, O, h).values
        E = feat_ok & np.isfinite(TGT)
        cut = t_start - h
        Etr = E.copy(); Etr[cut:] = False
        Y = np.nan_to_num((pd.DataFrame(np.where(Etr, TGT, np.nan)).rank(axis=1, pct=True) - 0.5).values.astype(np.float32))
        age = (cut - 1 - np.arange(T)).clip(min=0)
        w = np.repeat((0.5 ** (age / 365.0))[:, None], N, axis=1)[Etr]
        m = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
                             min_child_weight=50, reg_lambda=10.0, tree_method="hist", n_jobs=2, random_state=0)
        m.fit(Xcs[Etr], Y[Etr], sample_weight=w)
        rows = []
        for t in range(t_start, T - 1 - (h if h > 1 else 0)):
            cols = np.where(feat_ok[t] & np.isfinite(TGT[t]))[0]
            if len(cols) < 40:
                continue
            s = m.predict(Xcs[t, cols])
            top = cols[np.argsort(-s)[:10]]
            ex_oc = TGT[t, top].mean() - TGT[t, cols].mean()
            hh = max(h, 1)
            cch = (C.iloc[min(t + hh, T - 1)] / C.iloc[t] - 1).values
            ex_cc = np.nanmean(cch[top]) - np.nanmean(cch[cols])
            rows.append((dates[t].date(), ex_oc * 100, ex_cc * 100, float(np.nanmean(cch[cols]) * 100)))
        out += [f"## احتفاظ {h} يوم", "", "| يوم التوقع | excess oc | excess cc | السوق cc |", "|---|---|---|---|"]
        for d, a1, b1, c1 in rows:
            out.append(f"| {d} | {a1:+.2f} | {b1:+.2f} | {c1:+.2f} |")
        if rows:
            out.append(f"| **المتوسط** | **{np.mean([r[1] for r in rows]):+.2f}** | **{np.mean([r[2] for r in rows]):+.2f}** | |")
            out.append(f"\nجلسات أحسن من السوق (cc): {sum(r[2] > 0 for r in rows)} من {len(rows)}")
        out.append("")
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    open(os.path.join(bt.RES_DIR, "report_recent_check_latest.md"), "w", encoding="utf-8").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--start", default="2026-10-25")
    ap.add_argument("--horizons", type=lambda s: [int(x) for x in s.split(",") if x], default=[1, 5])
    a = ap.parse_args()
    try:
        run(a)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        open(os.path.join(bt.RES_DIR, "error_recent_check_latest.txt"), "w", encoding="utf-8").write(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
