#!/usr/bin/env python3
"""
================================================================
   EGX Multi-Day Holding Backtest - هل فيه ميزة لو احتفظت أكتر من يوم؟
================================================================
نفس منهج backtest.py (walk-forward، تكلفة، حكم بيبص على 2022+ لوحده) بس بفترات احتفاظ
3 و5 و10 و20 يوم بدل يوم واحد. الفكرة: التكلفة بتتوزع على فترة أطول، فده المكان الوحيد
اللي ممكن يطلع فيه مكسب بعد التكلفة لو اليومي مفيهوش.

التنفيذ الواقعي: الإشارة من قفلة اليوم t، شراء عند افتتاح t+1، بيع عند قفلة t+h.
  - من غير الأسهم اللي اتحركت 9%+ النهاردة (غالبًا قفلت على الحد)
  - التكلفة = رحلة كاملة (شراء + بيع) لكل صفقة، يعني متحفظة (لو سهم فضل في القايمة مش هتدفع تاني)
  - XGBoost بيتدرّب على الماضي بس، مع embargo: صفوف التدريب لازم نتيجتها تكون خلصت قبل التوقع،
    عشان مفيش تسريب من المستقبل في الفترات الأطول
  - الأرقام بالنسبة لمتوسط السوق في نفس اليوم، وبتتحول لـ bps/يوم عشان تتقارن بين الفترات

الاستخدام:
  python backtest/backtest_horizons.py --source yfinance
  python backtest/backtest_horizons.py --source synthetic-null | synthetic-planted   # اختبار سلامة
================================================================
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as bt  # noqa: E402
import macro  # noqa: E402

XF2 = bt.XF + ["r60", "r120"]


# ============================================================
# بيانات اصطناعية لاختبار السلامة (على فترات طويلة)
# ============================================================
def make_synthetic_h(planted, T=1700, N=150, seed=2):
    """planted=True: كل سهم عنده drift بطيء (نصف عمره ~25 يوم) فالعائد الماضي بيتنبأ بالمستقبل
    على فترة 10-20 يوم (momentum متوسط المدى). planted=False: مفيش أي إشارة."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-06-01", periods=T)
    m = rng.normal(0.0003, 0.008, T)
    beta = rng.uniform(0.5, 1.3, N)
    vol = rng.uniform(0.012, 0.03, N)
    noise = rng.normal(0, 1, (T, N)) * vol
    z = np.zeros((T, N))
    if planted:
        shock = rng.normal(0, 0.0009, (T, N))
        for t in range(1, T):
            z[t] = 0.97 * z[t - 1] + shock[t]
    r = np.clip(beta * m[:, None] + z + noise, -0.1, 0.1)
    C = 10 * np.cumprod(1 + r, axis=0)
    prev = np.vstack([C[:1], C[:-1]])
    opn = prev * (1 + rng.normal(0, 0.003, (T, N)))
    spread = np.abs(rng.normal(0, 0.01, (T, N)))
    cols = [f"S{i:03d}.CA" for i in range(N)]
    mk = lambda a: pd.DataFrame(a, index=dates, columns=cols)  # noqa: E731
    return {"Open": mk(opn), "High": mk(C * (1 + spread)), "Low": mk(C * (1 - spread)),
            "Close": mk(C), "Volume": mk(rng.lognormal(13, 0.5, (T, N)))}


# ============================================================
# الأهداف والتدريب والتقييم
# ============================================================
def build_target(C, O, h):
    """عائد الشراء عند افتتاح t+1 والبيع عند قفلة t+h. بيتشال أي صف فيه قفزة > 25% في يوم
    (تقسيم/بيانات غلط) أو فيه سعر ناقص في الدخول أو الخروج."""
    O1 = O.shift(-1)
    tgt = C.shift(-h) / O1 - 1
    ok = ((C.shift(-1) / O1 - 1).abs() <= bt.MAX_ABS_RET) & O1.notna() & C.shift(-h).notna()
    if h > 1:
        jump = ((C / C.shift(1) - 1).abs() > bt.MAX_ABS_RET).astype(float)
        jw = jump.rolling(h - 1, min_periods=h - 1).sum().shift(-h)   # قفزات الأيام t+2..t+h
        ok = ok & (jw == 0)
    return tgt.where(ok)


def run_xgb_h(Xcs, Y, E, first_valid, min_train, block, trees, embargo, seed=0):
    import xgboost as xgb
    T, N, K = Xcs.shape
    pred = np.full((T, N), np.nan)
    t0 = first_valid + min_train
    importance = None
    for b in range(t0, T - 1, block):
        cut = b - embargo               # صف t نتيجته بتخلص عند قفلة t+h، لازم تكون <= b
        tr = E[:cut]
        if tr.sum() < 5000:
            continue
        model = xgb.XGBRegressor(
            n_estimators=trees, max_depth=3, learning_rate=0.05, subsample=0.7,
            colsample_bytree=0.7, min_child_weight=50, reg_lambda=10.0,
            tree_method="hist", n_jobs=2, random_state=seed)
        model.fit(Xcs[:cut][tr], Y[:cut][tr])
        importance = model.feature_importances_
        for t in range(b, min(b + block, T - 1)):
            cols = np.where(np.isfinite(Xcs[t]).all(axis=1))[0]
            if len(cols):
                pred[t, cols] = model.predict(Xcs[t, cols])
    return pred, importance, t0


def evaluate_h(name, S, TGT, E, idxs, dates, cost, rng, min_cross, h):
    rows = []
    for t in idxs:
        if S is None:
            sc = rng.normal(size=E.shape[1])
            m = E[t]
        else:
            m = E[t] & np.isfinite(S[t])
            sc = S[t]
        n = int(m.sum())
        if n < min_cross:
            continue
        cols = np.where(m)[0]
        s = sc[cols] + rng.normal(0, 1e-9, n)
        y = TGT[t, cols]
        top = cols[np.argsort(-s)[: max(n // 5, 5)]]
        rows.append((dates[t], bt.rank_corr(s, y), TGT[t, top].mean() - y.mean()))
    if len(rows) < 60:
        return None
    d = pd.DataFrame(rows, columns=["date", "ic", "ex"]).set_index("date")
    rt = 2 * cost                                        # رحلة كاملة لكل صفقة
    d["ex_d"] = d["ex"] / h * 1e4                        # bps/يوم قبل التكلفة
    d["net_d"] = (d["ex"] - rt) / h * 1e4                # bps/يوم بعد التكلفة
    lag = h + 4                                          # الصفوف المتجاورة بتتشارك أيام (overlap)
    yearly = d.groupby(d.index.year).agg(days=("ic", "size"), ic=("ic", "mean"),
                                         ex_d=("ex_d", "mean"), net_d=("net_d", "mean"))
    yearly = yearly[yearly["days"] >= 100]
    pos_years = float((yearly["net_d"] > 0).mean()) if len(yearly) else float("nan")
    rec = d[d.index.year >= bt.RECENT_FROM]
    nan = float("nan")
    res = {
        "name": name, "h": h, "days": len(d),
        "ic": float(d["ic"].mean()), "ic_t": bt.nw_t(d["ic"].dropna(), lag),
        "ex_d": float(d["ex_d"].mean()), "net_d": float(d["net_d"].mean()),
        "pos_years": pos_years, "n_years": int(len(yearly)),
        "rec_days": int(len(rec)),
        "rec_ic": float(rec["ic"].mean()) if len(rec) else nan,
        "rec_ic_t": bt.nw_t(rec["ic"].dropna(), lag) if len(rec) >= 60 else nan,
        "rec_ex_d": float(rec["ex_d"].mean()) if len(rec) else nan,
        "rec_net_d": float(rec["net_d"].mean()) if len(rec) else nan,
    }
    rec_alive = bool(res["rec_days"] >= 100 and res["rec_ic_t"] == res["rec_ic_t"]
                     and res["rec_ic_t"] >= 2 and res["rec_ex_d"] > 0)
    full_sig = bool(res["ic_t"] >= 3 and res["ex_d"] > 0)
    if full_sig and rec_alive and res["net_d"] > 0 and res["rec_net_d"] > 0 \
            and pos_years == pos_years and pos_years >= 0.6:
        res["status"] = "worthy"
    elif full_sig and rec_alive:
        res["status"] = "gross_only"
    elif full_sig:
        res["status"] = "faded"
    else:
        res["status"] = "none"
    res["worthy"] = res["status"] == "worthy"
    return res, yearly


# ============================================================
# التقرير
# ============================================================
SORT = lambda r: -(r["rec_net_d"] if r["rec_net_d"] == r["rec_net_d"] else -9e9)  # noqa: E731


def fmt_report(horizons, meta):
    L = [f"# EGX Multi-Day Holding Backtest - {meta['run_at']}" + (" (بمدخلات EGX30 والدولار)" if meta.get("macro") else ""), "",
         f"- المصدر: **{meta['source']}** | الأسهم: {meta['tickers']} | الأيام: {meta['days_total']} "
         f"({meta['first']} → {meta['last']})",
         f"- تنفيذ واقعي: شراء عند افتتاح اليوم التالي للإشارة وبيع عند قفلة اليوم h، من غير الأسهم اللي "
         f"اتحركت 9%+ يوم الإشارة | تكلفة {meta['cost_bps']:.0f} bps في الاتجاه الواحد (رحلة كاملة لكل صفقة)",
         f"- كل الأرقام **bps/يوم** بالنسبة لمتوسط السوق، والحكم بيبص على {bt.RECENT_FROM}+ لوحدها كمان",
         "", "## الملخص: صافي bps/يوم بعد التكلفة في 2022+ (أعلى 20% من الأسهم)",
         "| الاستراتيجية | " + " | ".join(f"احتفاظ {h['h']} يوم" for h in horizons) + " |",
         "|---|" + "---|" * len(horizons)]
    names = []
    for h in horizons:
        for r in h["results"]:
            if r["name"] not in names:
                names.append(r["name"])
    for n in names:
        cells = []
        for h in horizons:
            r = next((x for x in h["results"] if x["name"] == n), None)
            cells.append(f"{r['rec_net_d']:+.1f} {bt.STATUS_ICON[r['status']]}" if r else "-")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    for h in horizons:
        L += ["", f"## احتفاظ {h['h']} يوم", f"فترة الاختبار: {h['oos_first']} → {h['oos_last']} ({h['oos_days']} يوم)", "",
              f"| الاستراتيجية | IC | t | صافي كل الفترة | IC {bt.RECENT_FROM}+ | t {bt.RECENT_FROM}+ | "
              f"قبل التكلفة {bt.RECENT_FROM}+ | صافي {bt.RECENT_FROM}+ | الحكم |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(h["results"], key=SORT):
            L.append(f"| {r['name']} | {r['ic']:+.4f} | {r['ic_t']:+.1f} | {r['net_d']:+.1f} | "
                     f"{r['rec_ic']:+.4f} | {r['rec_ic_t']:+.1f} | {r['rec_ex_d']:+.1f} | "
                     f"{r['rec_net_d']:+.1f} | {bt.STATUS_AR[r['status']]} |")
        best = sorted(h["results"], key=SORT)[0]
        yy = h["yearlies"][best["name"]]
        L += ["", f"سنة بسنة (صافي bps/يوم): {best['name']}",
              "| " + " | ".join(str(y) for y in yy.index) + " |", "|" + "---|" * len(yy),
              "| " + " | ".join(f"{v:+.1f}" for v in yy["net_d"]) + " |"]
        if h.get("importance") is not None:
            imp = sorted(zip(h.get("names") or XF2 + bt.MF, h["importance"]), key=lambda z: -z[1])[:8]
            L += ["", "أهم مدخلات XGBoost: " + ", ".join(f"{n} ({v:.2f})" for n, v in imp)]
    L += ["", "## حدود لازم تتحط في الاعتبار",
          "1. التكلفة رحلة كاملة لكل صفقة (متحفظة). بدون spread/slippage فعلي، وبيانات Open من Yahoo مش مثالية.",
          "2. الصفوف المتجاورة بتتشارك أيام، فالـ t-stat بيتحسب بتصحيح Newey-West (lag = h+4).",
          "3. صف بدون سعر دخول أو خروج، أو فيه قفزة > 25% في يوم، بيتشال (ممكن يحيّز النتايج شوية).",
          "4. قايمة الأسهم هي المقيدة حاليًا بس (survivorship bias). وسنين 2012-2017 كتير منها غايب لقلة الأسهم الصالحة.",
          "5. نتيجة موجبة = دليل مبدئي بس. الخطوة الجاية دايمًا تجربة حية قبل أي قرار فلوس."]
    return "\n".join(L)


def tg_summary(horizons):
    lines = ["🧪 باكتست الاحتفاظ لأكتر من يوم (تنفيذ واقعي، صافي بعد التكلفة، 2022+)", ""]
    for h in horizons:
        top = sorted(h["results"], key=SORT)[:3]
        lines.append(f"📅 احتفاظ {h['h']} يوم:")
        for r in top:
            lines.append(f"{bt.STATUS_ICON[r['status']]} {r['name']}: {r['rec_net_d']:+.1f} bps/يوم "
                         f"(قبل التكلفة {r['rec_ex_d']:+.1f}، IC {r['rec_ic']:+.3f}, t={r['rec_ic_t']:+.1f})")
        lines.append("")
    worthy = [f"{r['name']} ({h['h']} يوم)" for h in horizons for r in h["results"] if r["worthy"]]
    lines.append("تستحق تجربة حية: " + ("، ".join(worthy) if worthy else "لأ، ولا واحدة"))
    lines.append("✅ تستحق | 🟡 حية بس التكلفة بتاكلها | 🟠 اختفت حديثًا | ❌ مفيش ميزة")
    lines.append("التقرير الكامل: backtest/results/report_horizons_latest.md")
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def run(a):
    cost = a.cost_bps / 1e4
    rng = np.random.default_rng(42)
    horizons_req = [int(x) for x in a.horizons.split(",") if x.strip()]

    if a.source == "yfinance":
        panel = bt.fetch_yfinance(bt.load_tickers())
    else:
        panel = make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    if C.shape[1] < 20 or C.shape[0] < a.min_train + 150:
        raise bt.BacktestError(f"البيانات مش كفاية: {C.shape[1]} سهم × {C.shape[0]} يوم بعد التنضيف")
    print(f"[data] {C.shape[1]} سهم × {C.shape[0]} يوم ({C.index[0].date()} → {C.index[-1].date()})")

    f, mkt, _nxt, _nxt_oc, lock, liquid, flat59 = bt.build_features(C, H, L, V, O, a.min_turnover)
    lr = np.log1p(f["r1"])
    f["r60"] = np.expm1(lr.rolling(60, min_periods=60).sum())
    f["r120"] = np.expm1(lr.rolling(120, min_periods=120).sum())
    dates = C.index
    T, N = C.shape
    M = None
    if a.macro:
        M = macro.fetch_macro(fresh=True) if a.source == "yfinance" else macro.synthetic_macro(mkt["mkt1"])
        if M is None or not len(M.columns):
            raise bt.BacktestError("طلبت --macro بس Yahoo مرجّعش EGX30 ولا الدولار")
        print(f"[macro] مدخلات الاقتصاد الكلي: {', '.join(M.columns)}")
    Xcs, feat_names = macro.build_X(f, mkt, liquid, N, XF2, bt.MF, M, bt.csrank)
    print(f"[features] {len(feat_names)} مدخل: {', '.join(feat_names)}")
    feat_ok = np.isfinite(Xcs).all(axis=2) & liquid.values & ~lock.values

    cur = (f["tech"] + f["p_up"] / 1000.0).where(flat59 <= 0.25)
    rule = {"الموديل الحالي (تأكيد فني)": cur.values,
            "momentum 5 أيام": f["r5"].values, "momentum 20 يوم": f["r20"].values,
            "momentum 60 يوم": f["r60"].values, "momentum 120 يوم": f["r120"].values,
            "reversal 5 أيام": (-f["r5"]).values, "تذبذب منخفض": (-f["vol20"]).values,
            "عشوائي (اختبار سلامة)": None}

    out_h = []
    for h in horizons_req:
        TGT = build_target(C, O, h).values
        E = feat_ok & np.isfinite(TGT)
        elig = E.sum(axis=1)
        if elig.max() < 20:
            raise bt.BacktestError(f"احتفاظ {h} يوم: أقصى عدد أسهم صالحة = {int(elig.max())} "
                                   f"(جرّب تقلل min_turnover)")
        min_cross = int(np.clip(0.5 * np.percentile(elig[elig > 0], 75), 20, bt.MIN_CROSS))
        first_valid = int(np.where(elig >= min_cross)[0][0])
        Y = (pd.DataFrame(np.where(E, TGT, np.nan)).rank(axis=1, pct=True) - 0.5).values.astype(np.float32)
        print(f"\n[h={h}] min_cross={min_cross} | XGBoost walk-forward (embargo {h - 1} يوم)")
        pred, importance, t0 = run_xgb_h(Xcs, np.nan_to_num(Y), E, first_valid, a.min_train,
                                         a.block, a.trees, embargo=h - 1)
        idxs = [t for t in range(t0, T - 1) if np.isfinite(pred[t]).any()]
        if not idxs:
            raise bt.BacktestError(f"احتفاظ {h} يوم: مفيش تاريخ كفاية للاختبار خارج العينة")
        scores = dict(rule)
        scores["XGBoost (walk-forward)"] = pred
        results, yearlies = [], {}
        for name, S in scores.items():
            out = evaluate_h(name, S, TGT, E, idxs, dates, cost, rng, min_cross, h)
            if out:
                results.append(out[0])
                yearlies[name] = out[1]
                r = out[0]
                print(f"  {name:36s} IC {r['ic']:+.4f} (t={r['ic_t']:+.1f}) | صافي {r['net_d']:+.1f} bps/يوم | "
                      f"{bt.RECENT_FROM}+: IC {r['rec_ic']:+.4f} (t={r['rec_ic_t']:+.1f}) "
                      f"قبل {r['rec_ex_d']:+.1f} صافي {r['rec_net_d']:+.1f} | {bt.STATUS_ICON[r['status']]}")
        if not results:
            raise bt.BacktestError(f"احتفاظ {h} يوم: مفيش يوم فيه {min_cross}+ سهم صالح في فترة الاختبار")
        out_h.append({"h": h, "results": results, "yearlies": yearlies, "importance": importance,
                      "names": feat_names,
                      "oos_first": str(dates[idxs[0]].date()), "oos_last": str(dates[idxs[-1]].date()),
                      "oos_days": len(idxs)})

    meta = {"run_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "source": a.source,
            "tickers": N, "days_total": T, "first": str(dates[0].date()), "last": str(dates[-1].date()),
            "cost_bps": a.cost_bps, "min_tov": a.min_turnover, "block": a.block, "macro": bool(a.macro)}
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "horizons_latest" if a.source == "yfinance" else f"horizons_{a.source}"
    if a.macro:
        tag = tag.replace("horizons", "horizons_macro", 1)
    with open(os.path.join(bt.RES_DIR, f"report_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(fmt_report(out_h, meta))
    with open(os.path.join(bt.RES_DIR, f"results_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "horizons": {str(h["h"]): h["results"] for h in out_h}},
                  fh, ensure_ascii=False, indent=2)
    print(f"[done] التقرير: backtest/results/report_{tag}.md")
    if a.source == "yfinance" and not a.no_telegram:
        bt.send_telegram(tg_summary(out_h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance",
                    choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--cost-bps", type=float, default=15.0, help="تكلفة الاتجاه الواحد بالـ bps")
    ap.add_argument("--min-turnover", type=float, default=1_000_000)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--horizons", default="3,5,10,20")
    ap.add_argument("--macro", action="store_true",
                    help="ضيف مدخلات EGX30 والدولار (التقرير بيتحفظ باسم report_horizons_macro_latest.md للمقارنة)")
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()
    try:
        run(a)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        with open(os.path.join(bt.RES_DIR, "error_horizons_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M}\n{tb}")
        if a.source == "yfinance" and not a.no_telegram:
            kind = "فشل (متوقع)" if isinstance(e, bt.BacktestError) else "خطأ في الكود"
            bt.send_telegram(f"🧪❌ باكتست الاحتفاظ لأكتر من يوم وقع - {kind}:\n{type(e).__name__}: "
                             f"{str(e)[:600]}\nالتفاصيل في backtest/results/error_horizons_latest.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
