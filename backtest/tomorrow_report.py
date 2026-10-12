"""
================================================================
   EGX - تقرير جلسة بكره (ترتيب الأسهم المرشحة + النطاق المحتمل للسعر)
================================================================
بيشتغل بعد قفل السوق: بيدرّب نفس موديل XGBoost اللي اتقاس في الباكتست
(العالم ب: شراء عند الافتتاح وبيع عند القفل، من غير الأسهم اللي قفلت
على الحد) على كل التاريخ المتاح، وبيطلّع:

  1) أقوى الأسهم ترتيبًا لجلسة بكره (ترتيب نسبي بين الأسهم، مش نسبة احتمال)
  2) لكل سهم: آخر قفلة ونطاق السعر المعتاد لبكره (من حركة السهم نفسه
     في آخر 60 يوم، ومحصور بحد EGX اليومي ±10%)

⚠️ مهم: الباكتست بيقول إن ميزة اليوم الواحد صغيرة جدًا (IC حوالي +0.05)
وبعد التكلفة سالبة من 2022. فالتقرير ده ترتيب إحصائي ضعيف، مش توقع سعر
ولا توصية. النطاق وصف لحركة السهم المعتادة، مش هدف.

الاستخدام:
    python backtest/tomorrow_report.py                 # بيبعت تليجرام لو التوكن متحدد
    python backtest/tomorrow_report.py --no-telegram   # للتجربة بس
================================================================
"""

import argparse
import os
import sys
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import backtest as bt  # noqa: E402

OUT_DIR = os.path.join(ROOT, "tomorrow")
EGX_DAILY_LIMIT = 0.10        # حد EGX اليومي ±10%
RANGE_WINDOW = 60             # عدد الأيام اللي بنحسب منها نطاق الحركة المعتاد
TRADING_DAYS = {6, 0, 1, 2, 3}  # الأحد-الخميس (weekday(): الأحد=6)
AR_DAYS = {6: "الأحد", 0: "الاتنين", 1: "التلات", 2: "الأربع", 3: "الخميس", 4: "الجمعة", 5: "السبت"}


def next_trading_day(now):
    d = now.date()
    # لو لسه قبل الافتتاح في يوم تداول، "الجلسة الجاية" هي النهاردة
    if now.weekday() in TRADING_DAYS and now.hour < 10:
        return d
    d += timedelta(days=1)
    while d.weekday() not in TRADING_DAYS:
        d += timedelta(days=1)
    return d


def train_and_score(C, H, L, V, O, min_turnover, trees):
    """بيرجّع (scores, cols, names, f, lock, last_idx) لآخر يوم في البيانات."""
    import xgboost as xgb

    f, mkt, _nxt, nxt_oc, lock, liquid, _flat59 = bt.build_features(C, H, L, V, O, min_turnover)
    T, N = C.shape
    cs = {k: bt.csrank(f[k], liquid).values for k in bt.XF}
    Xcs = np.stack([cs[k] for k in bt.XF] +
                   [np.repeat(mkt[k].values[:, None], N, axis=1) for k in bt.MF], axis=2).astype(np.float32)
    feat_ok = np.isfinite(Xcs).all(axis=2) & liquid.values

    # التدريب: نفس العالم ب في الباكتست (افتتاح→قفلة لليوم التالي، من غير الأسهم المقفولة على الحد)
    E = feat_ok & ~lock.values & np.isfinite(nxt_oc.values)
    if int(E.sum()) < 5000:
        raise bt.BacktestError(f"بيانات التدريب قليلة ({int(E.sum())} صف)")
    Y = (pd.DataFrame(np.where(E, nxt_oc.values, np.nan)).rank(axis=1, pct=True) - 0.5).values
    model = xgb.XGBRegressor(
        n_estimators=trees, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
        min_child_weight=50, reg_lambda=10.0, tree_method="hist", n_jobs=2, random_state=0)
    model.fit(Xcs[E], Y[E].astype(np.float32))

    t = T - 1
    # الأسهم اللي مناسبة للتقرير: features كاملة + سيولة + مش قافلة على الحد النهاردة
    cols = np.where(feat_ok[t] & ~lock.values[t])[0]
    if len(cols) < 30:
        raise bt.BacktestError(f"عدد الأسهم الصالحة في آخر يوم قليل ({len(cols)})")
    score = model.predict(Xcs[t, cols])
    return score, cols, np.array(C.columns), f, lock, t


def price_range(C, col, t, window=RANGE_WINDOW):
    """النطاق المعتاد لقفلة بكره من توزيع حركات السهم نفسه في آخر `window` يوم."""
    px = C.iloc[:, col]
    last = float(px.iloc[t])
    r = (px / px.shift(1) - 1).iloc[max(0, t - window + 1):t + 1].dropna()
    r = r[r.abs() <= bt.MAX_ABS_RET]
    if len(r) < 20:
        return None
    lo, med, hi = np.percentile(r, [10, 50, 90])
    lo, hi = max(lo, -EGX_DAILY_LIMIT), min(hi, EGX_DAILY_LIMIT)
    return {"last": last, "low": last * (1 + lo), "mid": last * (1 + med), "high": last * (1 + hi),
            "lo_pct": lo * 100, "hi_pct": hi * 100,
            "ceil": last * (1 + EGX_DAILY_LIMIT), "floor": last * (1 - EGX_DAILY_LIMIT)}


def build_report(C, H, L, V, O, a, now):
    score, cols, names, f, lock, t = train_and_score(C, H, L, V, O, a.min_turnover, a.trees)
    data_date = C.index[t].date()
    target = next_trading_day(now)
    stale_days = (now.date() - data_date).days

    order = np.argsort(-score)
    n = len(order)
    top_k = min(a.top, n)
    rows = []
    for rank, i in enumerate(order[:top_k], 1):
        col = int(cols[i])
        pr = price_range(C, col, t)
        if pr is None:
            continue
        rows.append({"rank": rank, "name": names[col].replace(".CA", ""), "pr": pr,
                     "r1": float(f["r1"].iloc[t, col]) * 100 if np.isfinite(f["r1"].iloc[t, col]) else float("nan")})

    when = "النهاردة" if target == now.date() else "بكره"
    L_ = [f"🔮 تقرير جلسة {when} - {AR_DAYS[target.weekday()]} {target}",
          f"(مبني على بيانات قفلة {data_date} - {n} سهم سيولته كافية)", ""]
    if stale_days >= 2 and now.weekday() in TRADING_DAYS and now.hour >= 15:
        L_ += [f"⚠️ آخر بيانات عند Yahoo هي {data_date} (متأخرة {stale_days} يوم عن النهاردة). "
               f"الترتيب ده مبني على بيانات قديمة شوية.", ""]
    L_ += [f"📈 أقوى {len(rows)} سهم ترتيبًا لجلسة {when} (من الأقوى):",
           f"السهم | آخر قفلة | النطاق المعتاد ل{when} (80% من الأيام)", ""]
    for r in rows:
        p = r["pr"]
        L_.append(f"{r['rank']}. {r['name']} | {p['last']:.2f} | "
                  f"{p['low']:.2f} ← {p['mid']:.2f} → {p['high']:.2f}  "
                  f"({p['lo_pct']:+.1f}% / {p['hi_pct']:+.1f}%)")
    L_ += ["",
           "اللي في النص = الحركة الوسيطة المعتادة للسهم، مش هدف. الحد الأقصى النهاري لأي سهم ±10% من القفلة.",
           "",
           "⚠️ اقرأ ده قبل ما تعتمد على التقرير:",
           "• الترتيب ده ميزة إحصائية صغيرة جدًا في الباكتست (IC حوالي +0.05 ليوم واحد)، مش توقع إن السهم هيطلع.",
           "• بعد تكلفة الشراء والبيع (15 bps في الاتجاه) النتيجة في الباكتست كانت سالبة من 2022 لليوم الواحد.",
           "• ممكن أي سهم من القايمة ينزل بكره، والنطاق وصف للحركة الماضية مش ضمان.",
           "• مش توصية شراء ولا بيع. القرار والمسؤولية ليك."]
    return "\n".join(L_), rows, data_date


def send_long(text, limit=3800):
    chunk = ""
    for line in text.split("\n"):
        if len(chunk) + len(line) + 1 > limit and chunk:
            bt.send_telegram(chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        bt.send_telegram(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-turnover", type=float, default=1_000_000)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--top", type=int, default=10, help="عدد الأسهم في التقرير")
    ap.add_argument("--fresh", action="store_true", help="تجاهل كاش الأسعار وجيب بيانات جديدة")
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()
    now = datetime.now(ZoneInfo("Africa/Cairo"))
    try:
        if a.fresh:
            cache = os.path.join(bt.DATA_DIR, "prices.pkl")
            if os.path.exists(cache):
                os.remove(cache)
        panel = bt.fetch_yfinance(bt.load_tickers())
        C, H, L, V, O = bt.clean_panel(panel)
        # قبل القفل (2:45) أي صف بتاريخ النهاردة هو شمعة ناقصة، فبنشيله عشان الترتيب يتبني على آخر قفلة كاملة
        if now.weekday() in TRADING_DAYS and (now.hour, now.minute) < (14, 45):
            keep = C.index.date != now.date()
            C, H, L, V, O = (x[keep] for x in (C, H, L, V, O))
        text, rows, data_date = build_report(C, H, L, V, O, a, now)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "report_latest.md"), "w", encoding="utf-8") as fh:
            fh.write(text)
        # أي error قديم يتشال بعد نجاح التشغيل
        err = os.path.join(OUT_DIR, "error_latest.txt")
        if os.path.exists(err):
            os.remove(err)
        print(text)
        if not a.no_telegram:
            send_long(text)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "error_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M}\n{tb}")
        if not a.no_telegram:
            kind = "فشل (متوقع)" if isinstance(e, bt.BacktestError) else "خطأ في الكود"
            bt.send_telegram(f"🔮❌ تقرير جلسة بكره وقع - {kind}:\n{type(e).__name__}: {str(e)[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
