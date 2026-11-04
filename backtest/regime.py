#!/usr/bin/env python3
"""
تشخيص: ليه ترشيحات البوت قصرت عن السوق في الفترة اللي عدّت (6 من 7 جلسات أقل من متوسط السوق)؟
بيقيس على كل التاريخ، وبيقسّم حسب حالة السوق، ويقارن:
  S0 الموديل الحالي | HB أعلى بيتا (حساسية للسوق) | LB أقل بيتا | MIX متوسط ترتيب الموديل والبيتا
ومعاه: متوسط بيتا الأسهم اللي الموديل اختارها (هل بيختار دفاعي؟)
الأرقام: bps للصفقة (فرق أعلى 20% عن متوسط السوق)، قبل التكلفة، لأن الفرق بين الاستراتيجيات هو المهم.
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


def run(a):
    if a.source == "yfinance":
        panel = bt.fetch_yfinance(bt.load_tickers())
    else:
        panel = bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    dates, (T, N) = C.index, C.shape
    f, mkt, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, 1_000_000)
    lr = np.log1p(f["r1"])
    f["r60"] = np.expm1(lr.rolling(60, min_periods=60).sum())
    f["r120"] = np.expm1(lr.rolling(120, min_periods=120).sum())
    M = macro.fetch_macro(fresh=True) if a.source == "yfinance" else macro.synthetic_macro(mkt["mkt1"])
    M, egx_src = macro.ensure_egx30(M, C, V)
    Xcs, names = macro.build_X(f, mkt, liquid, N, bh.XF2, bt.MF, M, bt.csrank)
    bi = names.index("beta60")
    BETA = Xcs[:, :, bi]                                  # ترتيب نسبي بين الأسهم (0..1)
    feat_ok = np.isfinite(Xcs).all(axis=2) & liquid.values & ~lock.values
    mk20 = ((1 + mkt["mkt1"].fillna(0)).rolling(20).apply(np.prod, raw=True) - 1).values
    mk5 = ((1 + mkt["mkt1"].fillna(0)).rolling(5).apply(np.prod, raw=True) - 1).values
    regimes = {"السوق نازل (20 يوم ≤ -5%)": mk20 <= -0.05,
               "بعد نزول وبيرتد (20 يوم ≤ -5% و5 أيام ≥ +2%)": (mk20 <= -0.05) & (mk5 >= 0.02),
               "نازل ومكمل (20 يوم ≤ -5% و5 أيام ≤ -2%)": (mk20 <= -0.05) & (mk5 <= -0.02),
               "عادي": (mk20 > -0.05) & (mk20 < 0.05), "طالع (20 يوم ≥ +5%)": mk20 >= 0.05}
    out = [f"# EGX Regime - {datetime.now():%Y-%m-%d %H:%M}", "",
           "الأرقام bps للصفقة: متوسط (أعلى 20% - متوسط السوق)، قبل التكلفة. الأيام متداخلة، فالعدد مش عينات مستقلة.", ""]
    for h in a.horizons:
        print(f"[regime] h={h}")
        TGT = bh.build_target(C, O, h).values
        E = feat_ok & np.isfinite(TGT)
        elig = E.sum(axis=1)
        min_cross = int(np.clip(0.5 * np.percentile(elig[elig > 0], 75), 20, bt.MIN_CROSS))
        fv = int(np.where(elig >= min_cross)[0][0])
        Y = np.nan_to_num((pd.DataFrame(np.where(E, TGT, np.nan)).rank(axis=1, pct=True) - 0.5).values.astype(np.float32))
        pred, t0 = dg.walk_forward(Xcs, Y, E, fv, a.min_train, a.block, a.trees, max(h - 1, 0), dates, halflife=365)
        rk = pd.DataFrame(pred).rank(axis=1, pct=True).values
        scores = {"S0 الموديل": pred, "HB أعلى بيتا": BETA, "LB أقل بيتا": -BETA, "MIX موديل+بيتا": rk + BETA}
        acc = {k: {r: [] for r in regimes} for k in scores}
        beta_sel = {r: [] for r in regimes}
        for t in range(t0, T - 1):
            m = E[t] & np.isfinite(pred[t]) & np.isfinite(BETA[t])
            n = int(m.sum())
            if n < 30:
                continue
            cols = np.where(m)[0]
            y = TGT[t, cols]
            for k, S in scores.items():
                top = cols[np.argsort(-S[t, cols])[: max(n // 5, 5)]]
                ex = TGT[t, top].mean() - y.mean()
                for r, mask in regimes.items():
                    if mask[t]:
                        acc[k][r].append(ex)
                if k.startswith("S0"):
                    for r, mask in regimes.items():
                        if mask[t]:
                            beta_sel[r].append(float(BETA[t, top].mean()))
        out += [f"## احتفاظ {h} يوم", "", "| الحالة | أيام | " + " | ".join(scores) + " | متوسط بيتا اللي الموديل بيختاره (0 = دفاعي، 1 = عالي الحساسية) |",
                "|---|---|" + "---|" * len(scores) + "---|"]
        for r in regimes:
            n = len(acc["S0 الموديل"][r])
            if n < 20:
                continue
            cells = [f"{np.mean(acc[k][r]) * 1e4:+.0f}" for k in scores]
            out.append(f"| {r} | {n} | " + " | ".join(cells) + f" | {np.mean(beta_sel[r]):.2f} |")
        out.append("")
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "regime_latest" if a.source == "yfinance" else f"regime_{a.source}"
    with open(os.path.join(bt.RES_DIR, f"report_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--horizons", type=lambda s: [int(x) for x in s.split(",") if x], default=[1, 5, 10, 20])
    a = ap.parse_args()
    try:
        run(a)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        with open(os.path.join(bt.RES_DIR, "error_regime_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M}\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
