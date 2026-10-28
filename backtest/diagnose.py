#!/usr/bin/env python3
"""
================================================================
   EGX - تشخيص الموديل: بيغلط فين وإيه اللي ناقصه؟
================================================================
بيشغّل نفس XGBoost walk-forward بتاع الباكتست (احتفاظ h يوم، بمدخلات EGX30 والدولار) وبيقسّم نتيجته:
  A) حركة السوق كله: بعد يوم نزول/طلوع قوي، إيه اللي بيحصل بعده (ارتداد ولا كمل)؟
  B) الميزة (IC وعائد أعلى 20% عن السوق) حسب:
       - ظروف السوق (نازل / عادي / طالع، من تغير السوق في آخر 20 يوم)
       - سيولة السهم (تلات شرايح)
       - تذبذب السهم (تلات شرايح)
       - السنة
  C) تجربة وزن الأحدث أكتر (half-life) بدل ما كل السنين بنفس الوزن
  D) حساسية للتكلفة (15 و25 و40 bps في الاتجاه)
  E) اتجاه شراء/بيع (classifier): دقته حسب مستوى ثقته، مقابل التخمين الثابت

الاستخدام:
  python backtest/diagnose.py --source yfinance --horizon 10
  python backtest/diagnose.py --source synthetic-planted --horizon 10 --trees 30 --block 126   # تجربة سلامة
================================================================
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

RECENT = 2022


def walk_forward(Xcs, Y, E, first_valid, min_train, block, trees, embargo, dates, halflife=None, seed=0):
    """زي bh.run_xgb_h بس بيقبل halflife (بالأيام): وزن صف أقدم بـ halflife يوم = نص وزن الصف الأحدث."""
    import xgboost as xgb
    T, N, K = Xcs.shape
    pred = np.full((T, N), np.nan)
    t0 = first_valid + min_train
    for b in range(t0, T - 1, block):
        cut = b - embargo
        tr = E[:cut]
        if tr.sum() < 5000:
            continue
        model = xgb.XGBRegressor(
            n_estimators=trees, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
            min_child_weight=50, reg_lambda=10.0, tree_method="hist", n_jobs=2, random_state=seed)
        w = None
        if halflife:
            age = (cut - 1 - np.arange(cut)).astype(float)             # عمر الصف بالأيام
            w = np.repeat((0.5 ** (age / halflife))[:, None], N, axis=1)[tr]
        model.fit(Xcs[:cut][tr], Y[:cut][tr], sample_weight=w)
        for t in range(b, min(b + block, T - 1)):
            cols = np.where(np.isfinite(Xcs[t]).all(axis=1))[0]
            if len(cols):
                pred[t, cols] = model.predict(Xcs[t, cols])
    return pred, t0


def per_day(pred, TGT, E, idxs, groups=None):
    """لكل يوم: IC وفرق أعلى 20% عن متوسط الأسهم. groups: dict اسم -> مصفوفة (T,N) من أرقام شرايح 0/1/2."""
    rows = []
    for t in idxs:
        m = E[t] & np.isfinite(pred[t])
        n = int(m.sum())
        if n < 30:
            continue
        cols = np.where(m)[0]
        s, y = pred[t, cols], TGT[t, cols]
        top = cols[np.argsort(-s)[: max(n // 5, 5)]]
        rec = {"t": t, "ic": bt.rank_corr(s, y), "ex": float(TGT[t, top].mean() - y.mean())}
        if groups:
            for gname, G in groups.items():
                g = G[t, cols]
                for b in (0, 1, 2):
                    mb = g == b
                    if mb.sum() >= 15:
                        yb, sb = y[mb], s[mb]
                        tb = np.argsort(-sb)[: max(int(mb.sum()) // 5, 3)]
                        rec[f"{gname}{b}_ic"] = bt.rank_corr(sb, yb)
                        rec[f"{gname}{b}_ex"] = float(yb[tb].mean() - yb.mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def tercile(df):
    """شريحة 0/1/2 لكل صف (يوم) على مستوى الأسهم (cross-sectional)."""
    r = df.rank(axis=1, pct=True)
    return (np.floor(r.values * 3).clip(0, 2)).astype(float)


def summarize(d, dates, h, cost, label=""):
    rt = 2 * cost
    d = d.copy()
    d["date"] = [dates[t] for t in d["t"]]
    d["year"] = [x.year for x in d["date"]]
    rec = d[d["year"] >= RECENT]
    return {"label": label, "days": len(d), "rec_days": len(rec),
            "ic": float(d["ic"].mean()), "rec_ic": float(rec["ic"].mean()) if len(rec) else float("nan"),
            "gross_trade": float(rec["ex"].mean() * 1e4) if len(rec) else float("nan"),       # bps للصفقة
            "net_trade": float((rec["ex"].mean() - rt) * 1e4) if len(rec) else float("nan")}


def md_table(rows, cols, heads):
    L = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for r in rows:
        L.append("| " + " | ".join(c(r) if callable(c) else str(r[c]) for c in cols) + " |")
    return L


def run(a):
    cost = a.cost_bps / 1e4
    h = a.horizon
    if a.source == "yfinance":
        panel = bt.fetch_yfinance(bt.load_tickers())
    else:
        panel = bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    f, mkt, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, a.min_turnover)
    lr = np.log1p(f["r1"])
    f["r60"] = np.expm1(lr.rolling(60, min_periods=60).sum())
    f["r120"] = np.expm1(lr.rolling(120, min_periods=120).sum())
    dates = C.index
    T, N = C.shape

    M = macro.fetch_macro(fresh=True) if a.source == "yfinance" else macro.synthetic_macro(mkt["mkt1"])
    M, egx_src = macro.ensure_egx30(M, C, V)
    Xcs, names = macro.build_X(f, mkt, liquid, N, bh.XF2, bt.MF, M, bt.csrank)
    feat_ok = np.isfinite(Xcs).all(axis=2) & liquid.values & ~lock.values
    TGT = bh.build_target(C, O, h).values
    E = feat_ok & np.isfinite(TGT)
    elig = E.sum(axis=1)
    min_cross = int(np.clip(0.5 * np.percentile(elig[elig > 0], 75), 20, bt.MIN_CROSS))
    first_valid = int(np.where(elig >= min_cross)[0][0])
    Y = (pd.DataFrame(np.where(E, TGT, np.nan)).rank(axis=1, pct=True) - 0.5).values.astype(np.float32)
    Y = np.nan_to_num(Y)
    out = [f"# EGX Diagnose - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- الأسهم: {N} | الأيام: {T} ({dates[0].date()} → {dates[-1].date()}) | احتفاظ {h} يوم | "
           f"EGX30 من: {'Yahoo' if egx_src == 'yahoo' else 'مؤشر تقريبي'} | تكلفة {a.cost_bps:.0f} bps في الاتجاه",
           f"- الحكم بيبص على {RECENT}+ لوحدها. أرقام الصفقة بالـbps للصفقة الكاملة (مش للسهم/يوم).", ""]

    # ---------- A) حركة السوق كله
    mk = mkt["mkt1"].fillna(0)
    nxt1 = mk.shift(-1)
    nxt5 = (1 + mk).rolling(5).apply(np.prod, raw=True).shift(-5) - 1
    A = []
    for name, cond in (("كل الأيام", mk == mk), ("اليوم ≤ -3%", mk <= -0.03), ("اليوم ≤ -5%", mk <= -0.05),
                       ("اليوم ≥ +3%", mk >= 0.03), ("اليوم ≥ +5%", mk >= 0.05)):
        v1, v5 = nxt1[cond].dropna(), nxt5[cond].dropna()
        if len(v1) >= 3:
            A.append({"c": name, "n": len(v1), "p1": f"{(v1 > 0).mean() * 100:.0f}%",
                      "m1": f"{v1.mean() * 100:+.2f}%", "p5": f"{(v5 > 0).mean() * 100:.0f}%" if len(v5) else "-",
                      "m5": f"{v5.mean() * 100:+.2f}%" if len(v5) else "-"})
    out += ["## A) السوق كله بعد يوم حركته قوية (متوسط الأسهم)", "",
            *md_table(A, ["c", "n", "p1", "m1", "p5", "m5"],
                      ["الحالة", "عدد الأيام", "اليوم التالي طلع", "متوسط اليوم التالي", "5 أيام طلعت", "متوسط 5 أيام"]),
            "", "(لو بعد النزول القوي بيطلع غالبًا: السوق عنده ارتداد، والموديل محتاج مدخل بيشوفه.)", ""]

    # ---------- B) الموديل الأساسي
    print(f"[diag] walk-forward h={h} (بدون وزن)")
    pred, t0 = walk_forward(Xcs, Y, E, first_valid, a.min_train, a.block, a.trees, h - 1, dates)
    idxs = [t for t in range(t0, T - 1) if np.isfinite(pred[t]).any()]
    tov = (C * V).rolling(20, min_periods=15).median()
    G = {"liq": tercile(tov.where(liquid)), "vol": tercile(f["vol20"].where(liquid))}
    d0 = per_day(pred, TGT, E, idxs, G)
    d0["date"] = [dates[t] for t in d0["t"]]
    d0["year"] = [x.year for x in d0["date"]]
    mk20 = (1 + mk).rolling(20).apply(np.prod, raw=True) - 1
    d0["mk20"] = [float(mk20.iloc[t]) for t in d0["t"]]
    base = summarize(d0, dates, h, cost, "الأساسي")
    out += [f"## B) الموديل الأساسي (XGBoost + EGX30 والدولار)", "",
            f"- IC كل الفترة {base['ic']:+.4f} | IC {RECENT}+ {base['rec_ic']:+.4f} | "
            f"أعلى 20% عن السوق {RECENT}+: {base['gross_trade']:+.0f} bps للصفقة قبل التكلفة، {base['net_trade']:+.0f} بعدها", ""]
    rec = d0[d0["year"] >= RECENT]

    def grp_rows(frame, key_fn, labels):
        rows = []
        for lab, mask in labels:
            x = frame[mask(frame)]
            if len(x) >= 20:
                rows.append({"g": lab, "n": len(x), "ic": f"{x['ic'].mean():+.3f}",
                             "gross": f"{x['ex'].mean() * 1e4:+.0f}", "net": f"{(x['ex'].mean() - 2 * cost) * 1e4:+.0f}"})
        return rows

    out += ["### حسب ظروف السوق (تغير السوق في آخر 20 يوم) - " + f"{RECENT}+", "",
            *md_table(grp_rows(rec, None, [("السوق نازل (≤ -5%)", lambda x: x["mk20"] <= -0.05),
                                           ("عادي (-5% لـ +5%)", lambda x: (x["mk20"] > -0.05) & (x["mk20"] < 0.05)),
                                           ("السوق طالع (≥ +5%)", lambda x: x["mk20"] >= 0.05)]),
                      ["g", "n", "ic", "gross", "net"], ["الحالة", "أيام", "IC", "قبل التكلفة (bps/صفقة)", "بعد التكلفة"]), ""]
    for gname, title in (("liq", "سيولة السهم"), ("vol", "تذبذب السهم")):
        rows = []
        for b, lab in ((0, "الشريحة الأقل"), (1, "الوسط"), (2, "الشريحة الأعلى")):
            ic, ex = rec[f"{gname}{b}_ic"].dropna(), rec[f"{gname}{b}_ex"].dropna()
            if len(ex) >= 20:
                rows.append({"g": lab, "n": len(ex), "ic": f"{ic.mean():+.3f}",
                             "gross": f"{ex.mean() * 1e4:+.0f}", "net": f"{(ex.mean() - 2 * cost) * 1e4:+.0f}"})
        out += [f"### حسب {title} - {RECENT}+", "",
                *md_table(rows, ["g", "n", "ic", "gross", "net"],
                          ["الشريحة", "أيام", "IC", "قبل التكلفة (bps/صفقة)", "بعد التكلفة"]), ""]
    yr = d0.groupby("year").agg(n=("ic", "size"), ic=("ic", "mean"), ex=("ex", "mean"))
    yr = yr[yr["n"] >= 100]
    out += ["### سنة بسنة", "",
            *md_table([{"y": y, "n": int(r.n), "ic": f"{r.ic:+.3f}", "g": f"{r.ex * 1e4:+.0f}", "net": f"{(r.ex - 2 * cost) * 1e4:+.0f}"}
                       for y, r in yr.iterrows()], ["y", "n", "ic", "g", "net"],
                      ["السنة", "أيام", "IC", "قبل التكلفة (bps/صفقة)", "بعد التكلفة"]), ""]

    # ---------- C) وزن الأحدث
    ws = []
    for hl in a.halflives:
        print(f"[diag] walk-forward h={h} half-life={hl}")
        p2, _ = walk_forward(Xcs, Y, E, first_valid, a.min_train, a.block, a.trees, h - 1, dates, halflife=hl)
        i2 = [t for t in range(t0, T - 1) if np.isfinite(p2[t]).any()]
        s2 = summarize(per_day(p2, TGT, E, i2), dates, h, cost, f"half-life {hl} يوم")
        ws.append(s2)
    ws = [base] + ws
    out += ["## C) وزن الأحدث أكتر في التدريب", "",
            *md_table([{"l": r["label"], "ic": f"{r['rec_ic']:+.4f}", "g": f"{r['gross_trade']:+.0f}", "n": f"{r['net_trade']:+.0f}"} for r in ws],
                      ["l", "ic", "g", "n"], ["الإعداد", f"IC {RECENT}+", "قبل التكلفة (bps/صفقة)", "بعد التكلفة"]), ""]

    # ---------- D) حساسية التكلفة
    g = base["gross_trade"]
    out += ["## D) لو التكلفة الحقيقية أعلى (spread + ضريبة + slippage)", "",
            *md_table([{"c": c, "net": f"{g - 2 * c:+.0f}"} for c in (15, 25, 40)], ["c", "net"],
                      ["تكلفة الاتجاه الواحد (bps)", "صافي الصفقة (bps)"]), ""]

    # ---------- E) دقة اتجاه شراء/بيع حسب الثقة
    try:
        import xgboost as xgb
        cut = T - 250 - h
        Etr = E.copy(); Etr[cut:] = False
        Ete = np.zeros_like(E); Ete[T - 250:] = E[T - 250:]
        ytr = (TGT[Etr] > 0).astype(int)
        mdl = xgb.XGBClassifier(n_estimators=a.trees, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
                                min_child_weight=50, reg_lambda=10.0, tree_method="hist", n_jobs=2, random_state=0, eval_metric="logloss")
        mdl.fit(Xcs[Etr], ytr)
        p = mdl.predict_proba(Xcs[Ete])[:, 1]
        y = (TGT[Ete] > 0).astype(int)
        up = y.mean()
        bins = [("p ≥ 0.60 (واثق إنه يطلع)", p >= 0.60), ("0.55 ≤ p < 0.60", (p >= 0.55) & (p < 0.60)),
                ("0.45 < p < 0.55 (مش واثق)", (p > 0.45) & (p < 0.55)), ("0.40 < p ≤ 0.45", (p > 0.40) & (p <= 0.45)),
                ("p ≤ 0.40 (واثق إنه ينزل)", p <= 0.40)]
        rows = []
        for lab, m in bins:
            if m.sum() >= 30:
                rows.append({"b": lab, "n": int(m.sum()), "up": f"{y[m].mean() * 100:.0f}%"})
        out += [f"## E) اتجاه شراء/بيع: آخر 250 يوم (مدرّب على اللي قبلهم بس)", "",
                f"- نسبة الأسهم اللي طلعت فعلاً في الفترة دي: {up * 100:.0f}% | دقة 0.5 كحد: {((p >= 0.5) == y).mean() * 100:.0f}%", "",
                *md_table(rows, ["b", "n", "up"], ["مستوى الثقة", "عدد", "نسبة اللي طلعوا فعلاً"]), "",
                "(لو نسبة الطلوع بتزيد مع الثقة: التصنيف مفيد وممكن نخلي المنطقة الوسط 'انتظر'. لو ثابتة: مفيش ثقة حقيقية.)", ""]
    except Exception as e:  # noqa: BLE001
        out += [f"## E) فشل: {str(e)[:150]}", ""]

    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "diagnose_latest" if a.source == "yfinance" else f"diagnose_{a.source}"
    with open(os.path.join(bt.RES_DIR, f"report_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=15.0)
    ap.add_argument("--min-turnover", type=float, default=1_000_000)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--halflives", type=lambda s: [int(x) for x in s.split(",") if x], default=[365, 730])
    a = ap.parse_args()
    try:
        run(a)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        with open(os.path.join(bt.RES_DIR, "error_diagnose_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M}\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
