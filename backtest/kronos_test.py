#!/usr/bin/env python3
"""
اختبار Kronos (Foundation model للشموع، AAAI 2026) zero-shot على أسهم EGX، بنفس مقياس موديلنا:
لكل يوم اختبار (كل 5 جلسات في آخر ~200 جلسة): Kronos بياخد آخر 256 شمعة يومية للسهم وبيولّد 10 شموع جاية،
والترتيب = العائد المتوقع للقفلة بعد 10 جلسات. بنقارنه بـXGBoost (نفس الأيام والأسهم) ومتوسط ترتيبهم،
بالفرق عن متوسط السوق (أعلى 20%) بعد تكلفة 20 bps في الاتجاه.
ملاحظة: عينة صغيرة (عدد الأيام قليل لأن الـinference على CPU بطيء)، فهي مؤشر مش حكم.
"""
import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as bt  # noqa: E402
import backtest_horizons as bh  # noqa: E402
import diagnose as dg  # noqa: E402
import improve as im  # noqa: E402
import macro  # noqa: E402

LOOKBACK = 256


class StubPredictor:
    """للتجربة بس (من غير Kronos): بيرجّع مسار عشوائي."""
    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False):
        rng = np.random.default_rng(0)
        out = []
        for d, yt in zip(df_list, y_timestamp_list):
            last = float(d["close"].iloc[-1])
            path = last * np.cumprod(1 + rng.normal(0, 0.01, pred_len))
            out.append(pd.DataFrame({"close": path}, index=list(yt)))
        return out


def load_predictor(a):
    if a.stub:
        return StubPredictor()
    kd = os.environ.get("KRONOS_DIR", os.path.join(HERE, "..", "Kronos"))
    sys.path.insert(0, kd)
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained(f"NeoQuasar/Kronos-{a.size}")
    return KronosPredictor(mdl, tok, device="cpu", max_context=512)


def run(a):
    cost = a.cost_bps / 1e4
    h = a.horizon
    panel = bt.fetch_yfinance(bt.load_tickers()) if a.source == "yfinance" else bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    dates, (T, N) = C.index, C.shape
    f0, mkt0, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, 1_000_000)
    M = macro.fetch_macro(fresh=True) if a.source == "yfinance" else macro.synthetic_macro(mkt0["mkt1"])
    M, _src = macro.ensure_egx30(M, C, V)
    Xcs, feat_ok = im.setup(C, H, L, V, O, 1_000_000, M)
    TGT = bh.build_target(C, O, h).values
    E = feat_ok & np.isfinite(TGT)
    elig = E.sum(axis=1)
    mc = int(np.clip(0.5 * np.percentile(elig[elig > 0], 75), 20, bt.MIN_CROSS))
    fv = int(np.where(elig >= mc)[0][0])
    Y = np.nan_to_num((pd.DataFrame(np.where(E, TGT, np.nan)).rank(axis=1, pct=True) - 0.5).values.astype(np.float32))
    print("[kronos] XGBoost walk-forward...")
    xp, _t0 = dg.walk_forward(Xcs, Y, E, fv, a.min_train, a.block, a.trees, h - 1, dates, halflife=365)

    pr = load_predictor(a)
    Cf = C.ffill(limit=3)
    Of, Hf, Lf, Vf = (x.reindex_like(C).ffill(limit=3) for x in (O, H, L, V))
    ts = [T - 2 - h - a.step * i for i in range(a.n_dates)]
    ts = [t for t in sorted(ts) if t - LOOKBACK >= 0]
    results = []
    t_start = time.time()
    for t in ts:
        cols = [j for j in np.where(E[t] & np.isfinite(xp[t]))[0]
                if Cf.iloc[t - LOOKBACK + 1:t + 1, j].notna().all() and Of.iloc[t - LOOKBACK + 1:t + 1, j].notna().all()]
        if len(cols) < 40:
            continue
        cols = cols[: a.max_stocks]
        xt = pd.Series(dates[t - LOOKBACK + 1:t + 1])
        yt = pd.Series(pd.bdate_range(dates[t] + pd.Timedelta(days=1), periods=h))
        dfs = []
        for j in cols:
            d = pd.DataFrame({"open": Of.iloc[t - LOOKBACK + 1:t + 1, j].values, "high": Hf.iloc[t - LOOKBACK + 1:t + 1, j].values,
                              "low": Lf.iloc[t - LOOKBACK + 1:t + 1, j].values, "close": Cf.iloc[t - LOOKBACK + 1:t + 1, j].values,
                              "volume": Vf.iloc[t - LOOKBACK + 1:t + 1, j].fillna(0).values})
            d["amount"] = d["close"] * d["volume"]
            dfs.append(d)
        score = np.full(len(cols), np.nan)
        for b0 in range(0, len(dfs), a.batch):
            part = dfs[b0:b0 + a.batch]
            try:
                outp = pr.predict_batch(df_list=part, x_timestamp_list=[xt] * len(part), y_timestamp_list=[yt] * len(part),
                                        pred_len=h, T=1.0, top_p=0.9, sample_count=a.samples, verbose=False)
                for k, (d, o) in enumerate(zip(part, outp)):
                    score[b0 + k] = float(o["close"].iloc[-1]) / float(d["close"].iloc[-1]) - 1
            except Exception as e:  # noqa: BLE001
                print(f"[kronos] batch فشل: {str(e)[:200]}")
        ok = np.isfinite(score)
        if ok.sum() < 40:
            continue
        cc = np.array(cols)[ok]
        results.append((t, cc, score[ok]))
        print(f"[kronos] {dates[t].date()} | {int(ok.sum())} سهم | {time.time() - t_start:.0f}s")
        if time.time() - t_start > a.max_seconds:
            print("[kronos] وقفت بسبب حد الوقت")
            break

    def ev(fn):
        ex, ic = [], []
        for t, cc, ks in results:
            s = fn(t, cc, ks)
            y = TGT[t, cc]
            top = np.argsort(-s)[: max(len(cc) // 5, 5)]
            ex.append(float(y[top].mean() - y.mean()))
            ic.append(bt.rank_corr(s, y))
        ex = np.array(ex)
        if not len(ex):
            return None
        return {"n": len(ex), "gross": ex.mean() * 1e4, "net": (ex.mean() - 2 * cost) * 1e4, "ic": float(np.mean(ic)),
                "hit": float((ex > 2 * cost).mean() * 100)}

    def rk(v):
        return pd.Series(v).rank(pct=True).values
    variants = [("Kronos zero-shot", lambda t, cc, ks: ks),
                ("XGBoost (موديلنا الحالي)", lambda t, cc, ks: xp[t, cc]),
                ("متوسط ترتيب Kronos + XGBoost", lambda t, cc, ks: rk(ks) + rk(xp[t, cc]))]
    out = [f"# EGX Kronos Test - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- Kronos-{a.size} zero-shot (من غير تدريب على EGX) | احتفاظ {h} يوم | عينة: {len(results)} يوم اختبار (كل {a.step} جلسات) | "
           f"تكلفة {a.cost_bps:.1f} bps | أعلى 20% مقابل متوسط السوق", "",
           "| الطريقة | أيام | IC | قبل التكلفة (bps) | بعد التكلفة (bps) | أيام كسبانة |", "|---|---|---|---|---|---|"]
    for name, fn in variants:
        r = ev(fn)
        out.append(f"| {name} | {r['n']} | {r['ic']:+.3f} | {r['gross']:+.0f} | {r['net']:+.0f} | {r['hit']:.0f}% |" if r else f"| {name} | - |")
    out += ["", "⚠️ عينة صغيرة (الأيام متداخلة لأن الاحتفاظ 10 جلسات)، فالفروق الصغيرة ضجيج."]
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "kronos_latest" if a.source == "yfinance" and not a.stub else "kronos_stub"
    open(os.path.join(bt.RES_DIR, f"report_{tag}.md"), "w", encoding="utf-8").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--size", default="small", choices=["mini", "small", "base"])
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--n-dates", type=int, default=40)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--max-stocks", type=int, default=150)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--max-seconds", type=int, default=9000)
    a = ap.parse_args()
    try:
        run(a)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        open(os.path.join(bt.RES_DIR, "error_kronos_latest.txt"), "w", encoding="utf-8").write(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
