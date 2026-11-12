#!/usr/bin/env python3
"""هل دمج الموديل مع low-vol (أسهم هادية) بيحسّن؟ صفقات مش متداخلة 2022+، بعد التكلفة."""
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
import final_test as ft  # noqa: E402
import improve as im  # noqa: E402
import improve2 as i2  # noqa: E402
import macro  # noqa: E402


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
    lr = np.log(C / C.shift(1)).clip(-0.3, 0.3)
    vol = lr.rolling(60, min_periods=40).std()
    lv_rank = (-vol).rank(axis=1, pct=True).values            # أعلى = أهدى
    mom = (C / C.shift(120) - 1).rank(axis=1, pct=True).values
    P, tg, t0s = {}, {}, []
    for h in (20, 40, 60):
        print(f"[blend] h={h}")
        TGT, E, fv, Y = i2.prep(C, O, h, ok2)
        p, t0 = i2.walk("lgbm", X2, Y, E, fv, a, h - 1)
        P[h] = pd.DataFrame(p).rank(axis=1, pct=True).values
        tg[h] = (TGT, E)
        t0s.append(t0)
    t0 = max(t0s)
    ens = np.nanmean(np.stack([P[20], P[40], P[60]]), axis=0)
    out = [f"# EGX Blend Test - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- LightGBM+Alpha158 مقابل low-vol (تذبذب 60 يوم الأقل) ومقابل دمجهم. تكلفة {a.cost_bps:.1f} bps في الاتجاه، 2022+، صفقات مش متداخلة", ""]
    head = ft_head()
    variants = {
        "موديل (ensemble 20/40/60)": ens,
        "low-vol بس": lv_rank,
        "دمج 50/50 (موديل + low-vol)": 0.5 * ens + 0.5 * lv_rank,
        "دمج 70/30 (موديل + low-vol)": 0.7 * ens + 0.3 * lv_rank,
        "دمج موديل + low-vol + زخم 6 شهور (50/30/20)": 0.5 * ens + 0.3 * lv_rank + 0.2 * mom,
    }
    for h in (20, 40, 60):
        TGT, E = tg[h]
        out += [f"## احتفاظ {h} يوم", "", *head]
        for name, sc in variants.items():
            sc = np.where(np.isfinite(sc), sc, np.nan)
            for fr in (0.1, 0.2):
                out.append(ft.row(f"{name} - أعلى {fr * 100:.0f}%", ft.summarize(sc, TGT, E, t0, dates, h, fr, cost)))
        out.append("")
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "blend_latest" if a.source == "yfinance" else f"blend_{a.source}"
    open(os.path.join(bt.RES_DIR, f"report_{tag}.md"), "w", encoding="utf-8").write(text)
    print(text)


def ft_head():
    return ["| الإعداد | صفقات مستقلة | زيادة (bps) | t | صفقات غلبت السوق | عائد مطلق صافي | زيادة لما السوق طالع | زيادة لما السوق نازل | أسوأ صفقة (زيادة) |",
            "|---|---|---|---|---|---|---|---|---|"]


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
        open(os.path.join(bt.RES_DIR, "error_blend_latest.txt"), "w", encoding="utf-8").write(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
