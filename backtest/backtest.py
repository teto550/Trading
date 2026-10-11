#!/usr/bin/env python3
"""
================================================================
   EGX Walk-Forward Backtest - هل فيه إشارة حقيقية لتوقع اليوم التالي؟
================================================================
بيقارن الموديل الحالي (monitor_once.compute_outlook) ببدائل (reversal, momentum,
XGBoost) على تاريخ سنين من yfinance، بتقسيم زمني صارم (walk-forward):
  - كل إشارة بتتحسب من بيانات لحد قفلة اليوم t بس، والنتيجة هي عائد t -> t+1
  - XGBoost بيتدرّب على الماضي بس وبيتعاد تدريبه كل block يوم (expanding window)
  - بنقيس عائد أعلى 20% من الأسهم (وأعلى 5 أسهم) *بالنسبة لمتوسط السوق في نفس اليوم*،
    قبل وبعد تكاليف التداول، مع t-stat (Newey-West) واستقرار عبر السنين

القرار: الإشارة "تستحق تجربة حية" بس لو t-stat للـ IC >= 3 (عشان نعوّض تجربة
أكتر من استراتيجية) والعائد بعد التكاليف موجب وموجب في 60% من السنين على الأقل.

الاستخدام:
  python backtest/backtest.py --source yfinance            # الحقيقي (GitHub Actions)
  python backtest/backtest.py --source synthetic-null      # اختبار سلامة: مفيش إشارة
  python backtest/backtest.py --source synthetic-planted   # اختبار سلامة: إشارة مزروعة

⚠️ حدود: القفلة للقفلة (مفيش slippage/spread فعلي)، قايمة الأسهم هي الأسهم المقيدة
حاليًا (survivorship bias: الأسهم اللي اتشطبت مش موجودة)، وجودة بيانات Yahoo للسوق
المصري مش مثالية. النتيجة الإيجابية دليل مبدئي مش ضمان.
================================================================
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(HERE, "data")
RES_DIR = os.path.join(HERE, "results")

MAX_ABS_RET = 0.25  # حد EGX اليومي 10% - أي قفزة أكبر غالبًا تقسيم/بيانات غلط
MIN_HISTORY = 300   # أقل عدد أيام للسهم عشان يدخل الاختبار
MIN_CROSS = 60      # أقل عدد أسهم صالحة في اليوم عشان نقيّم اليوم ده

XF = ["r1", "r2", "r3", "r5", "r10", "r20", "vol20", "sg20", "sg50",
      "z1", "vr", "flat20", "rng5", "tech", "p_up"]  # features مرتبة cross-sectional
MF = ["mkt1", "breadth"]                               # features على مستوى السوق


# ============================================================
# البيانات
# ============================================================
def load_tickers():
    path = os.path.join(ROOT, "tickers.txt")
    raw = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = "\n".join(l for l in f if not l.lstrip().startswith("#"))
    seen, out = set(), []
    for t in re.findall(r"[A-Za-z0-9]{2,8}\.CA", raw):
        t = t.upper()
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def fetch_yfinance(tickers):
    import yfinance as yf
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "prices.pkl")
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 3 * 86400:
        print(f"[data] بستخدم الكاش: {cache}")
        return pd.read_pickle(cache)
    frames = {}
    for i, t in enumerate(tickers, 1):
        for attempt in range(3):
            try:
                h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=True)
                if len(h) > 0:
                    idx = pd.to_datetime(h.index).tz_localize(None).normalize()
                    h = h.copy()
                    h.index = idx
                    h = h[~h.index.duplicated(keep="last")]
                    frames[t] = h[["High", "Low", "Close", "Volume"]]
                break
            except Exception as e:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
                if attempt == 2:
                    print(f"  [فشل] {t}: {e}")
        if i % 25 == 0:
            print(f"[data] {i}/{len(tickers)} سهم، اتجاب تاريخ لـ {len(frames)}")
        time.sleep(0.3)
    if not frames:
        raise RuntimeError("مفيش بيانات اتجابت من yfinance")
    panel = {k: pd.DataFrame({t: f[k] for t, f in frames.items()})
             for k in ("High", "Low", "Close", "Volume")}
    pd.to_pickle(panel, cache)
    return panel


def make_synthetic(planted, T=1600, N=150, seed=1):
    """بيانات عشوائية: planted=True بتزرع reversal ضعيف (autocorrelation سالب)
    عشان نتأكد إن الباكتست بيلقطه، وplanted=False مفيهاش أي إشارة."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=T)
    m = rng.normal(0.0003, 0.008, T)
    beta = rng.uniform(0.5, 1.3, N)
    vol = rng.uniform(0.01, 0.03, N)
    noise = rng.normal(0, 1, (T, N)) * vol
    phi = 0.05 if planted else 0.0
    e = np.zeros((T, N))
    for t in range(1, T):
        e[t] = noise[t] - phi * e[t - 1]
    r = np.clip(beta * m[:, None] + e, -0.1, 0.1)
    C = 10 * np.cumprod(1 + r, axis=0)
    spread = np.abs(rng.normal(0, 0.01, (T, N)))
    cols = [f"S{i:03d}.CA" for i in range(N)]
    mk = lambda a: pd.DataFrame(a, index=dates, columns=cols)  # noqa: E731
    return {"High": mk(C * (1 + spread)), "Low": mk(C * (1 - spread)), "Close": mk(C),
            "Volume": mk(rng.lognormal(13, 0.5, (T, N)))}


def clean_panel(panel, min_cross=30):
    C, H, L, V = (panel[k].sort_index() for k in ("Close", "High", "Low", "Volume"))
    C = C.where(C > 0)
    keep_dates = C.notna().sum(axis=1) >= min_cross  # بيشيل الإجازات وأيام ماحدش اتداول فيها
    C, H, L, V = (x[keep_dates] for x in (C, H, L, V))
    ok = C.notna().sum() >= MIN_HISTORY
    C, H, L, V = (x.loc[:, ok] for x in (C, H, L, V))
    return C, H, L, V


# ============================================================
# Features (كلها بتستخدم بيانات لحد اليوم t بس)
# ============================================================
def build_features(C, H, L, V, min_tov):
    r = C / C.shift(1) - 1
    r = r.where(r.abs() <= MAX_ABS_RET)            # قفزات التقسيم/البيانات الغلط تتشال
    lr = np.log1p(r)
    P = np.exp(lr.fillna(0).cumsum()).where(C.notna())  # سعر معدّل من العوائد النضيفة
    f = {"r1": r}
    for k in (2, 3, 5, 10, 20):
        f[f"r{k}"] = np.expm1(lr.rolling(k, min_periods=k).sum())
    f["vol20"] = r.rolling(20, min_periods=15).std()
    sma20 = P.rolling(20, min_periods=20).mean()
    sma50 = P.rolling(50, min_periods=50).mean()
    f["sg20"] = P / sma20 - 1
    f["sg50"] = P / sma50 - 1
    f["z1"] = r / f["vol20"].replace(0, np.nan)
    vm = V.rolling(20, min_periods=15).mean().replace(0, np.nan)
    f["vr"] = V / vm
    f["flat20"] = (r == 0).where(r.notna()).rolling(20, min_periods=15).mean()
    f["rng5"] = ((H - L) / C).rolling(5, min_periods=4).mean()
    # نفس منطق monitor_once.compute_outlook بالظبط
    checks = ((P > sma20).astype(float) + (P > sma50) + (sma20 > sma50)
              + (P > P.shift(5)) + (P > P.shift(20)))
    valid_t = sma50.notna() & P.shift(20).notna()
    f["tech"] = (checks / 5 * 100).where(valid_t)
    ups = (r > 0).where(r.notna()).rolling(59, min_periods=30).sum()
    downs = (r < 0).where(r.notna()).rolling(59, min_periods=30).sum()
    f["p_up"] = (ups + 10) / (ups + downs + 20) * 100
    flat59 = (r == 0).where(r.notna()).rolling(59, min_periods=30).mean()
    tov = (C * V).rolling(20, min_periods=15).median()
    liquid = tov >= min_tov
    nxt = r.shift(-1)                                # عائد اليوم التالي (الهدف)
    mkt = {"mkt1": r.mean(axis=1), "breadth": (r > 0).where(r.notna()).mean(axis=1)}
    return f, mkt, nxt, liquid, flat59


def csrank(df, liquid):
    return df.where(liquid).rank(axis=1, pct=True) - 0.5


# ============================================================
# الاستراتيجيات
# ============================================================
def rule_scores(f, flat59):
    cur = f["tech"] + f["p_up"] / 1000.0   # الموديل الحالي: تأكيد فني أولاً ثم احتمال الصعود
    cur = cur.where(flat59 <= 0.25)         # نفس فلتر الأسهم الواقفة في collect_candidates
    return {
        "الموديل الحالي (تأكيد فني)": cur,
        "reversal 1 يوم (اشتري اللي نزل امبارح)": -f["r1"],
        "reversal 5 أيام": -f["r5"],
        "reversal معدّل بالتذبذب (z)": -f["z1"],
        "momentum 5 أيام": f["r5"],
        "momentum 20 يوم": f["r20"],
        "عشوائي (اختبار سلامة)": None,
    }


def run_xgb(Xcs, Y, E, first_valid, min_train, block, trees, seed=0):
    import xgboost as xgb
    T, N, K = Xcs.shape
    pred = np.full((T, N), np.nan)
    t0 = first_valid + min_train
    importance = None
    for b in range(t0, T - 1, block):
        tr = E[:b]                      # تدريب لحد b-1 بس (target اليوم b-1 معروف عند قفلة b)
        if tr.sum() < 5000:
            continue
        Xtr = Xcs[:b][tr]
        ytr = Y[:b][tr]
        model = xgb.XGBRegressor(
            n_estimators=trees, max_depth=3, learning_rate=0.05, subsample=0.7,
            colsample_bytree=0.7, min_child_weight=50, reg_lambda=10.0,
            tree_method="hist", n_jobs=2, random_state=seed)
        model.fit(Xtr, ytr)
        importance = model.feature_importances_
        end = min(b + block, T - 1)
        for t in range(b, end):
            cols = np.where(np.isfinite(Xcs[t]).all(axis=1))[0]
            if len(cols):
                pred[t, cols] = model.predict(Xcs[t, cols])
    return pred, importance, t0


# ============================================================
# التقييم
# ============================================================
def nw_t(x, lag=5):
    x = np.asarray(x, float)
    n = len(x)
    if n < 30:
        return float("nan")
    xm = x - x.mean()
    s = (xm * xm).sum() / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (xm[l:] * xm[:-l]).sum() / n
    se = np.sqrt(max(s, 0) / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def rank_corr(a, b):
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(name, S, NXT, E, idxs, dates, cost, rng):
    rows = []
    prev20 = prev5 = None
    for t in idxs:
        if S is None:
            sc = rng.normal(size=E.shape[1])
            m = E[t]
        else:
            m = E[t] & np.isfinite(S[t])
            sc = S[t]
        n = int(m.sum())
        if n < MIN_CROSS:
            continue
        cols = np.where(m)[0]
        s = sc[cols] + rng.normal(0, 1e-9, n)      # كسر التعادل عشوائيًا (مفيش تحيّز لترتيب الأعمدة)
        y = NXT[t, cols]
        order = np.argsort(-s)
        k = max(n // 5, 5)
        top20, top5 = cols[order[:k]], cols[order[:5]]
        mk = y.mean()
        t20 = 1.0 if prev20 is None else 1 - len(set(top20) & prev20) / len(top20)
        t5 = 1.0 if prev5 is None else 1 - len(set(top5) & prev5) / len(top5)
        prev20, prev5 = set(top20), set(top5)
        rows.append((dates[t], rank_corr(s, y),
                     NXT[t, top20].mean() - mk, NXT[t, top5].mean() - mk,
                     t20 * 2 * cost, t5 * 2 * cost))
    if len(rows) < 30:
        return None
    d = pd.DataFrame(rows, columns=["date", "ic", "ex20", "ex5", "c20", "c5"]).set_index("date")
    d["net20"] = d["ex20"] - d["c20"]
    d["net5"] = d["ex5"] - d["c5"]
    yearly = d.groupby(d.index.year).agg(days=("ic", "size"), ic=("ic", "mean"),
                                         ex20=("ex20", "mean"), net20=("net20", "mean"))
    yearly = yearly[yearly["days"] >= 100]
    pos_years = float((yearly["net20"] > 0).mean()) if len(yearly) else float("nan")
    res = {
        "name": name, "days": len(d),
        "ic": float(d["ic"].mean()), "ic_t": nw_t(d["ic"].dropna()),
        "ex20_bps": float(d["ex20"].mean() * 1e4), "ex20_t": nw_t(d["ex20"]),
        "net20_bps": float(d["net20"].mean() * 1e4),
        "hit20": float((d["ex20"] > 0).mean()),
        "ex5_bps": float(d["ex5"].mean() * 1e4), "net5_bps": float(d["net5"].mean() * 1e4),
        "turn20": float(d["c20"].mean() / (2 * cost)) if cost > 0 else float("nan"),
        "pos_years": pos_years, "n_years": int(len(yearly)),
    }
    res["worthy"] = bool(res["ic_t"] >= 3 and res["net20_bps"] > 0
                         and pos_years == pos_years and pos_years >= 0.6)
    # إشارة حقيقية إحصائيًا بس التكلفة (دوران يومي) بتاكل المكسب = محتاجة استراتيجية أبطأ
    gross_only = bool(not res["worthy"] and res["ic_t"] >= 3 and res["ex20_bps"] > 0)
    res["status"] = "worthy" if res["worthy"] else ("gross_only" if gross_only else "none")
    return res, yearly


# ============================================================
# التقرير
# ============================================================
STATUS_AR = {"worthy": "✅ تستحق تجربة حية",
             "gross_only": "🟡 إشارة بس التكلفة بتاكلها",
             "none": "❌ مفيش ميزة مؤكدة"}
STATUS_ICON = {"worthy": "✅", "gross_only": "🟡", "none": "❌"}


def fmt_report(results, yearlies, meta, importance):
    L = [f"# EGX Walk-Forward Backtest - {meta['run_at']}", "",
         f"- المصدر: **{meta['source']}** | الأسهم: {meta['tickers']} | الأيام: {meta['days_total']} "
         f"({meta['first']} → {meta['last']})",
         f"- فترة الاختبار خارج العينة: {meta['oos_first']} → {meta['oos_last']} "
         f"({meta['oos_days']} يوم)",
         f"- تكلفة الصفقة: {meta['cost_bps']:.0f} bps في الاتجاه الواحد | حد السيولة: "
         f"{meta['min_tov']:,.0f} جنيه/يوم | XGBoost: إعادة تدريب كل {meta['block']} يوم",
         "- كل الأرقام **بالنسبة لمتوسط السوق في نفس اليوم** (القفلة للقفلة)", "",
         "| الاستراتيجية | IC | t(IC) | أعلى 20%: bps/يوم | بعد التكلفة | يوم مكسب | أعلى 5 بعد التكلفة | سنين موجبة | الحكم |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: -(x["ic_t"] if x["ic_t"] == x["ic_t"] else -9)):
        py = f"{r['pos_years'] * 100:.0f}% ({r['n_years']})" if r["pos_years"] == r["pos_years"] else "-"
        L.append(f"| {r['name']} | {r['ic']:+.4f} | {r['ic_t']:+.2f} | {r['ex20_bps']:+.1f} | "
                 f"{r['net20_bps']:+.1f} | {r['hit20'] * 100:.0f}% | {r['net5_bps']:+.1f} | {py} | "
                 f"{STATUS_AR[r['status']]} |")
    L += ["", "**القاعدة:** ✅ = t(IC) ≥ 3 (بدل 2، لأننا جربنا أكتر من استراتيجية) + عائد موجب بعد "
          "التكاليف + موجب في 60% من السنين على الأقل. 🟡 = الإشارة حقيقية إحصائيًا قبل التكلفة بس "
          "الدوران اليومي بياكل المكسب (محتاجة احتفاظ أطول أو سيولة أعلى). ❌ = مفيش ميزة مؤكدة. "
          "IC = ارتباط ترتيب الإشارة بترتيب عائد اليوم التالي بين الأسهم؛ 0.02-0.03 يعتبر قوي."]
    if importance is not None:
        imp = sorted(zip(XF + MF, importance), key=lambda z: -z[1])[:8]
        L += ["", "## أهم مدخلات XGBoost (آخر تدريب)",
              ", ".join(f"{n} ({v:.2f})" for n, v in imp)]
    best = max(results, key=lambda x: x["ic_t"] if x["ic_t"] == x["ic_t"] else -9)
    L += ["", f"## تفصيل سنة بسنة: {best['name']}",
          "| السنة | أيام | IC | أعلى 20% bps | بعد التكلفة |", "|---|---|---|---|---|"]
    for yr, row in yearlies[best["name"]].iterrows():
        L.append(f"| {yr} | {int(row['days'])} | {row['ic']:+.4f} | {row['ex20'] * 1e4:+.1f} | "
                 f"{row['net20'] * 1e4:+.1f} |")
    L += ["", "## حدود لازم تتحط في الاعتبار",
          "1. القفلة للقفلة: مفيش spread ولا slippage، والتنفيذ الفعلي بيكون على سعر الافتتاح.",
          "2. قايمة الأسهم هي المقيدة حاليًا فقط (survivorship bias بيحسّن النتايج شوية).",
          "3. بيانات Yahoo للسوق المصري فيها فجوات وأسهم واقفة؛ بنشيل القفزات > 25% والأسهم قليلة السيولة.",
          "4. نتيجة موجبة هنا = دليل مبدئي. الخطوة الجاية دايمًا تجربة حية (scorecard) قبل أي قرار فلوس."]
    return "\n".join(L)


def tg_summary(results):
    worthy = [r for r in results if r["status"] == "worthy"]
    lines = ["🧪 نتيجة الباكتست (walk-forward على تاريخ EGX)"]
    for r in sorted(results, key=lambda x: -(x["ic_t"] if x["ic_t"] == x["ic_t"] else -9)):
        lines.append(f"{STATUS_ICON[r['status']]} {r['name']}: IC {r['ic']:+.3f} "
                     f"(t={r['ic_t']:+.1f}) | بعد التكلفة {r['net20_bps']:+.0f} bps/يوم")
    lines.append("")
    lines.append("فيه إشارة تستحق تجربة حية: " + ("، ".join(r["name"] for r in worthy)
                                                  if worthy else "لأ، ولا واحدة"))
    lines.append("🟡 = الإشارة حقيقية بس التكلفة بتاكلها، ❌ = مفيش ميزة مؤكدة")
    lines.append("التقرير الكامل في backtest/results/ على GitHub.")
    return "\n".join(lines)


def send_telegram(msg):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
    if not tok or not chat:
        return
    try:
        import requests
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": msg}, timeout=15)
    except Exception as e:  # noqa: BLE001
        print("[تليجرام] خطأ:", e)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance",
                    choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--cost-bps", type=float, default=15.0, help="تكلفة الاتجاه الواحد بالـ bps")
    ap.add_argument("--min-turnover", type=float, default=1_000_000, help="حد السيولة (جنيه/يوم)")
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()
    cost = a.cost_bps / 1e4
    rng = np.random.default_rng(42)

    if a.source == "yfinance":
        panel = fetch_yfinance(load_tickers())
    else:
        panel = make_synthetic(a.source == "synthetic-planted")
    C, H, L, V = clean_panel(panel)
    print(f"[data] {C.shape[1]} سهم × {C.shape[0]} يوم ({C.index[0].date()} → {C.index[-1].date()})")

    f, mkt, nxt, liquid, flat59 = build_features(C, H, L, V, a.min_turnover)
    dates = C.index
    T, N = C.shape
    NXT = nxt.values
    cs = {k: csrank(f[k], liquid).values for k in XF}
    Xcs = np.stack([cs[k] for k in XF] +
                   [np.repeat(mkt[k].values[:, None], N, axis=1) for k in MF], axis=2).astype(np.float32)
    E = np.isfinite(Xcs).all(axis=2) & np.isfinite(NXT) & liquid.values
    first_valid = int(np.argmax(E.sum(axis=1) >= MIN_CROSS))
    # هدف XGBoost: ترتيب عائد اليوم التالي بين الأسهم (cross-sectional rank)
    Y = (pd.DataFrame(np.where(E, NXT, np.nan)).rank(axis=1, pct=True) - 0.5).values.astype(np.float32)

    print(f"[xgb] walk-forward: تدريب أولي {a.min_train} يوم، إعادة تدريب كل {a.block} يوم")
    pred, importance, t0 = run_xgb(Xcs, np.nan_to_num(Y), E, first_valid, a.min_train, a.block, a.trees)
    idxs = [t for t in range(t0, T - 1) if np.isfinite(pred[t]).any()]
    if not idxs:
        sys.exit("مفيش تاريخ كفاية للاختبار خارج العينة (قلل --min-train أو استخدم بيانات أطول)")

    strategies = rule_scores(f, flat59)
    scores = {k: (None if v is None else v.values) for k, v in strategies.items()}
    scores["XGBoost (walk-forward)"] = pred

    results, yearlies = [], {}
    for name, S in scores.items():
        out = evaluate(name, S, NXT, E, idxs, dates, cost, rng)
        if out:
            results.append(out[0])
            yearlies[name] = out[1]
            r = out[0]
            print(f"  {name:42s} IC {r['ic']:+.4f} (t={r['ic_t']:+.2f}) | أعلى20% {r['ex20_bps']:+.1f} bps "
                  f"| بعد التكلفة {r['net20_bps']:+.1f} | {STATUS_ICON[r['status']]}")

    meta = {"run_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "source": a.source,
            "tickers": N, "days_total": T, "first": str(dates[0].date()), "last": str(dates[-1].date()),
            "oos_first": str(dates[idxs[0]].date()), "oos_last": str(dates[idxs[-1]].date()),
            "oos_days": len(idxs), "cost_bps": a.cost_bps, "min_tov": a.min_turnover, "block": a.block}
    report = fmt_report(results, yearlies, meta, importance)
    os.makedirs(RES_DIR, exist_ok=True)
    tag = "latest" if a.source == "yfinance" else a.source
    with open(os.path.join(RES_DIR, f"report_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(RES_DIR, f"results_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": results}, fh, ensure_ascii=False, indent=2)
    print(f"[done] التقرير: backtest/results/report_{tag}.md")
    if a.source == "yfinance" and not a.no_telegram:
        send_telegram(tg_summary(results))


if __name__ == "__main__":
    main()
