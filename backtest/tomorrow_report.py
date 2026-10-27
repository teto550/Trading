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
import json
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
import direction as dr  # noqa: E402
import news as nw  # noqa: E402

OUT_DIR = os.path.join(ROOT, "tomorrow")
FORECAST_DIR = os.path.join(ROOT, "forecast")   # توقع كل الأسهم (البوت بيقرا منه لما تكتب اسم سهم)
PICKS_LOG = os.path.join(OUT_DIR, "picks_log.json")    # ترشيحات كل جلسة (بتتسجل وقت التقرير)
SCORE_FILE = os.path.join(OUT_DIR, "scorecard.json")   # نتيجة كل جلسة اتقيّمت فعلاً
SESSION_CLOSE = (14, 30)      # قفل EGX 2:30 م بتوقيت القاهرة
EGX_DAILY_LIMIT = 0.10        # حد EGX اليومي ±10%
RANGE_WINDOW = 60             # عدد الأيام اللي بنحسب منها نطاق الحركة المعتاد
TRADING_DAYS = {6, 0, 1, 2, 3}  # الأحد-الخميس (weekday(): الأحد=6)
AR_DAYS = {6: "الأحد", 0: "الاتنين", 1: "التلات", 2: "الأربع", 3: "الخميس", 4: "الجمعة", 5: "السبت"}


def next_trading_day(now):
    d = now.date()
    # لو لسه قبل القفل في يوم تداول، "الجلسة الجاية" هي النهاردة (GitHub cron بيتأخر ساعات أحيانًا،
    # فتقرير الصبح المتأخر لازم يفضل متسجل لجلسة النهاردة مش بكره)
    if now.weekday() in TRADING_DAYS and (now.hour, now.minute) < SESSION_CLOSE:
        return d
    d += timedelta(days=1)
    while d.weekday() not in TRADING_DAYS:
        d += timedelta(days=1)
    return d


def extend_with_state(C, H, L, V, O, now, state_path=None, min_frac=0.7):
    """Yahoo بيتأخر جلسة عن EGX، لكن state.json (من Mubasher) فيه قفلة النهاردة.
    بنضيف صف لآخر يوم بس، وبنحسب تغير اليوم من state.json نفسه (مصدر واحد، فمفيش فرق
    تعديلات توزيعات بين المصدرين) ونطبقه على آخر قفلة معدّلة من Yahoo.
    القفلة هي المعلومة الحقيقية الوحيدة؛ المدى (H-L) والحجم تقديريين.
    بيرجّع (C, H, L, V, O, ملاحظة أو None)."""
    import json
    state_path = state_path or os.path.join(ROOT, "state.json")
    if not os.path.exists(state_path):
        return C, H, L, V, O, None
    with open(state_path, "r", encoding="utf-8") as fh:
        stocks = json.load(fh).get("stocks", {})
    last_y = C.index[-1]
    cand = [s["dates"][-1] for s in stocks.values() if s.get("dates")]
    if not cand:
        return C, H, L, V, O, None
    s_date = pd.Timestamp(max(cand))
    if s_date <= last_y:
        return C, H, L, V, O, None            # Yahoo مواكب أو أحدث، مفيش داعي
    cairo_today = pd.Timestamp(now.date())
    if s_date > cairo_today:
        return C, H, L, V, O, None
    if s_date == cairo_today and now.weekday() in TRADING_DAYS and (now.hour, now.minute) < (14, 45):
        return C, H, L, V, O, None            # الجلسة شغالة: سعر state لسه مش قفلة
    ys, ss = last_y.strftime("%Y-%m-%d"), s_date.strftime("%Y-%m-%d")
    new_c = {}
    for t in C.columns:
        st = stocks.get(t)
        if not st or ss not in st["dates"] or ys not in st["dates"]:
            continue
        p_new = st["daily"][st["dates"].index(ss)]
        p_old = st["daily"][st["dates"].index(ys)]
        y_last = C[t].iloc[-1]
        if p_old and p_old > 0 and p_new and p_new > 0 and np.isfinite(y_last):
            new_c[t] = float(y_last * p_new / p_old)
    if len(new_c) < min_frac * int(C.iloc[-1].notna().sum()):
        return C, H, L, V, O, None            # تغطية ناقصة: أأمن نفضل على Yahoo
    row = pd.Series(new_c).reindex(C.columns)
    prev = C.iloc[-1]
    vol = V.iloc[-20:].median()
    extra = {
        "C": row,
        "H": pd.concat([prev, row], axis=1).max(axis=1),
        "L": pd.concat([prev, row], axis=1).min(axis=1),
        "V": vol.where(row.notna()),
        "O": pd.Series(np.nan, index=C.columns),   # NaN: عشان تارجت امبارح (افتتاح→قفلة) ميتلوّثش
    }
    out = []
    for name, df in zip("CHLVO", (C, H, L, V, O)):
        r = extra[name].to_frame(s_date).T
        r.index = pd.DatetimeIndex([s_date])
        out.append(pd.concat([df, r]))
    note = (f"ℹ️ قفلة {ss} اتضافت من Mubasher ({len(new_c)} سهم) لأن Yahoo متأخر. "
            f"القفلة حقيقية، لكن مدى اليوم وحجم التداول تقديريين.")
    return (*out, note)


def train_and_score(C, H, L, V, O, min_turnover, trees):
    """بيرجّع (scores, cols, names, f, lock, last_idx, احتمال_الصعود, دقة_التصنيف) لآخر يوم في البيانات."""
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
    # شراء/بيع: موديل تاني بيجاوب "العائد المطلق بكره هيبقى موجب؟" + قياس دقته على آخر 250 يوم
    ret = nxt_oc.values
    dirp, hold = None, None
    try:
        dirp = dr.fit_predict(Xcs, ret, E, t, cols, trees)
        hold = dr.holdout(Xcs, ret, E, 1, trees)
    except Exception as e:  # noqa: BLE001
        print(f"[direction] فشل تصنيف شراء/بيع، هكمّل من غيره: {e}")
    return score, cols, np.array(C.columns), f, lock, t, dirp, hold


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


def fmt_px(x):
    return f"{x:.3f}" if x < 1 else f"{x:.2f}"


def levels(C, H, L, col, t):
    """مستويات فنية للسهم من بيانات لحد اليوم t: ATR14 وRSI14 ودعم/مقاومة آخر 20 يوم، ووقف خسارة
    وهدف مبنيين على تذبذب السهم، ونطاق أسبوع معتاد. دي مستويات للتخطيط، مش توقع إن الهدف هيتحقق."""
    if t < 125:
        return None
    c, h, l = C.iloc[:, col], H.iloc[:, col], L.iloc[:, col]
    last = float(c.iloc[t])
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.iloc[t - 13:t + 1].mean())
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(last) or last <= 0:
        return None
    d = c.diff().iloc[t - 13:t + 1]
    gain, loss = float(d.clip(lower=0).mean()), float((-d.clip(upper=0)).mean())
    rsi = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    support = float(l.iloc[t - 19:t + 1].min())
    resist = float(h.iloc[t - 19:t + 1].max())
    stop = max(last - 1.5 * atr, last * (1 - EGX_DAILY_LIMIT))
    target, capped = last + 3 * atr, False
    if last * 1.005 < resist < target:        # مقاومة قريبة قبل الهدف: الهدف الواقعي هو المقاومة
        target, capped = resist, True
    rr = (target - last) / (last - stop) if last > stop else float("nan")
    r5 = (c / c.shift(5) - 1).iloc[t - 119:t + 1].dropna()
    r5 = r5[r5.abs() <= bt.MAX_ABS_RET]
    if len(r5) < 40:
        return None
    w_lo, w_hi = np.percentile(r5, [10, 90])
    return {"last": last, "atr_pct": atr / last * 100, "rsi": rsi, "support": support, "resist": resist,
            "stop": stop, "target": target, "capped": capped, "rr": rr,
            "w_low": last * (1 + w_lo), "w_high": last * (1 + w_hi)}


def market_status(C, t):
    """حالة السوق كله: متوسط تغير اليوم ونسبة الأسهم الطالعة وتغير آخر 20 جلسة (وسيط الأسهم)."""
    r1 = (C.iloc[t] / C.iloc[t - 1] - 1)
    r1 = r1[r1.abs() <= bt.MAX_ABS_RET].dropna()
    r20 = (C.iloc[t] / C.iloc[t - 20] - 1)
    r20 = r20[r20.abs() <= 1.0].dropna()
    if len(r1) < 30 or len(r20) < 30:
        return None
    breadth, mean1, med20 = float((r1 > 0).mean()), float(r1.mean()), float(r20.median())
    if breadth <= 0.25 or med20 <= -0.05:
        tag, weak = "🔴 السوق نازل بقوة", True
    elif med20 >= 0.03 and breadth >= 0.5:
        tag, weak = "🟢 السوق طالع", False
    else:
        tag, weak = "🟡 السوق متذبذب", False
    return {"tag": tag, "weak": weak, "breadth": breadth * 100, "mean1": mean1 * 100, "med20": med20 * 100}


def build_report(C, H, L, V, O, a, now, note=None):
    score, cols, names, f, lock, t, dirp, hold = train_and_score(C, H, L, V, O, a.min_turnover, a.trees)
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
        rows.append({"rank": rank, "name": names[col].replace(".CA", ""), "pr": pr, "lv": levels(C, H, L, col, t),
                     "label": dr.label(float(dirp[i])) if dirp is not None else None,
                     "r1": float(f["r1"].iloc[t, col]) * 100 if np.isfinite(f["r1"].iloc[t, col]) else float("nan")})

    # توقع كل الأسهم الصالحة (للبوت: اكتب اسم سهم)
    all_stocks = {}
    for rank, i in enumerate(order, 1):
        col = int(cols[i])
        pr_ = price_range(C, col, t)
        lv_ = levels(C, H, L, col, t)
        all_stocks[names[col].replace(".CA", "")] = {
            "rank": rank, "last": float(C.iloc[t, col]),
            "label": dr.label(float(dirp[i])) if dirp is not None else None,
            "lo": round(pr_["low"], 4) if pr_ else None, "hi": round(pr_["high"], 4) if pr_ else None,
            "lv": ({k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) and v == v else None)
                    for k, v in lv_.items() if k in ("rsi", "atr_pct", "support", "resist", "stop", "target", "rr")}
                   if lv_ else None)}
    forecast = {"data_date": str(data_date), "target": str(target), "n": n, "hold": hold, "stocks": all_stocks}

    # أخبار الأسهم العشرة (تأكيد/تحذير بس، مبتغيّرش الترتيب ولا التصنيف)
    news_map = {}
    if not getattr(a, "no_news", False) and rows:
        news_map = nw.news_for([r["name"] for r in rows])
    for r in rows:
        info = news_map.get(r["name"])
        r["news"] = info
        r["news_ag"] = nw.agreement(r.get("label"), info["sent"]) if info else None
    if news_map:
        rows = nw.reorder(rows, news_map, lambda r: r["name"], lambda r: r.get("label"))

    when = "النهاردة" if target == now.date() else "بكره"
    L_ = [f"🔮 تقرير جلسة {when} - {AR_DAYS[target.weekday()]} {target}",
          f"(مبني على بيانات قفلة {data_date} - {n} سهم سيولته كافية)"]
    if note:
        L_.append(note)
    L_.append("")
    if stale_days >= 2 and now.weekday() in TRADING_DAYS and now.hour >= 15:
        L_ += [f"⚠️ آخر بيانات عند Yahoo هي {data_date} (متأخرة {stale_days} يوم عن النهاردة). "
               f"الترتيب ده مبني على بيانات قديمة شوية.", ""]
    ms = market_status(C, t)
    if ms:
        L_ += [f"{ms['tag']}: النهاردة {ms['mean1']:+.1f}% | {ms['breadth']:.0f}% من الأسهم طلعت | "
               f"آخر 20 جلسة {ms['med20']:+.1f}% (وسيط الأسهم)"]
        if ms["weak"]:
            L_ += ["⚠️ السوق نازل بقوة: الترشيحات دي أضعف من العادي، والدخول عكس اتجاه السوق خطره أعلى. "
                   "الأحسن تتابع وتستنى بدل ما تدخل."]
        L_.append("")
    L_ += [f"📈 أقوى {len(rows)} سهم ترتيبًا لجلسة {when} + مستوياتهم الفنية:"]
    if news_map:
        L_.append("(مرتبين: اللي الخبر بيأكد تحليله الأول، وبعده اللي الخبر عكسه، وبعده الباقي. "
                  "رقم ترتيب الموديل الأصلي جنب كل سهم)")
    L_.append("")
    for pos, r in enumerate(rows, 1):
        p, lv = r["pr"], r["lv"]
        lab = f" | {'🟢' if r['label'] == dr.BUY else '🔴'} {r['label']}" if r.get("label") else ""
        mr = f" [ترتيب الموديل {r['rank']}]" if news_map else ""
        L_.append(f"{pos if news_map else r['rank']}. {r['name']} | آخر قفلة {fmt_px(p['last'])}{lab}{mr}")
        if lv:
            L_.append(f"   RSI {lv['rsi']:.0f} | تذبذب يومي {lv['atr_pct']:.1f}% | "
                      f"دعم {fmt_px(lv['support'])} | مقاومة {fmt_px(lv['resist'])}")
            tag = " (عند المقاومة)" if lv["capped"] else ""
            weak_rr = " ⚠️ضعيف" if lv["rr"] < 1.5 else ""
            L_.append(f"   وقف خسارة {fmt_px(lv['stop'])} ({(lv['stop'] / lv['last'] - 1) * 100:+.1f}%) | "
                      f"هدف {fmt_px(lv['target'])} ({(lv['target'] / lv['last'] - 1) * 100:+.1f}%){tag} | "
                      f"عائد/مخاطرة {lv['rr']:.1f}{weak_rr}")
            L_.append(f"   نطاق {when}: {fmt_px(p['low'])} - {fmt_px(p['high'])} | "
                      f"نطاق الأسبوع: {fmt_px(lv['w_low'])} - {fmt_px(lv['w_high'])}")
        else:
            L_.append(f"   نطاق {when}: {fmt_px(p['low'])} - {fmt_px(p['high'])}")
        L_ += nw.news_lines(r.get("news"), r.get("label"))
        L_.append("")
    if news_map and nw.status_note():
        L_ += [nw.status_note(), ""]
    if news_map:
        L_ += ["📰 الأخبار تأكيد أو تحذير بس: مبتغيّرش الترتيب ولا شراء/بيع، ومفيش باكتست ليها لسه. "
               "أنا بقيس بعد كام جلسة لو التصنيف بيبقى أصح لما الأخبار بتأيده.", ""]
    if dirp is not None:
        L_ += ["🟢 شراء = الموديل شايف السهم هيقفل بكره فوق افتتاحه | 🔴 بيع = شايفه هيقفل تحته.",
               "⚠️ " + dr.holdout_line(hold), ""]
    L_ += ["إزاي تقرأ المستويات:",
           "• وقف الخسارة = 1.5 × التذبذب اليومي تحت القفلة. الهدف = 3 × التذبذب فوقها (أو المقاومة لو أقرب).",
           "• عائد/مخاطرة = كام هتكسب لو وصل الهدف مقابل كام هتخسر لو ضرب الوقف. أقل من 1.5 يبقى مش مجزي.",
           "• النطاقات وصف لحركة السهم الماضية (80% من الأيام)، والحد الأقصى النهاري ±10%.",
           "",
           "⚠️ اقرأ ده قبل ما تعتمد على التقرير:",
           "• الترتيب ميزة إحصائية صغيرة جدًا في الباكتست (IC حوالي +0.05 ليوم واحد)، مش توقع إن السهم هيطلع.",
           "• بعد تكلفة الشراء والبيع (15 bps في الاتجاه) النتيجة في الباكتست كانت سالبة من 2022 لليوم الواحد.",
           "• الوقف والهدف لإدارة الخسارة مش لضمان الوصول. أي سهم ممكن ينزل بكره.",
           "• مش توصية شراء ولا بيع. القرار والمسؤولية ليك."]
    return "\n".join(L_), rows, data_date, forecast



# ============================================================
# سجل الترشيحات + نتيجتها الفعلية (طلعوا ولا لأ)
# ============================================================
def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return default


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


def record_picks(rows, target, data_date, now):
    """بيسجّل أسهم التقرير لجلسة `target`. آخر تقرير لنفس الجلسة هو اللي بيتحسب،
    إلا لو الجلسة دي اتقيّمت خلاص فمبنغيّرش حاجة."""
    names = [r["name"] for r in rows]
    if not names:
        return
    scored = {d["target"] for d in _load_json(SCORE_FILE, {"days": []})["days"]}
    if str(target) in scored:
        return
    log = _load_json(PICKS_LOG, {"entries": []})
    entry = {"target": str(target), "data_date": str(data_date), "picks": names,
             "labels": {r["name"]: r["label"] for r in rows if r.get("label")},
             "news": {r["name"]: r["news_ag"] for r in rows if r.get("news_ag")},
             "made": now.strftime("%Y-%m-%d %H:%M")}
    ent = [e for e in log["entries"] if e["target"] != entry["target"]] + [entry]
    log["entries"] = sorted(ent, key=lambda e: e["target"])[-250:]
    _save_json(PICKS_LOG, log)


def score_pending(C):
    """بيقيّم كل جلسة مسجّل ليها ترشيحات وقفلتها موجودة في البيانات: التغير من قفلة الجلسة اللي
    قبلها لقفلة الجلسة نفسها، مقارنة بحركة السوق كله. بيرجّع قايمة الأيام الجديدة."""
    log = _load_json(PICKS_LOG, {"entries": []})
    sc = _load_json(SCORE_FILE, {"days": []})
    done = {d["target"] for d in sc["days"]}
    idx = {str(d.date()): i for i, d in enumerate(C.index)}
    cnt = C.notna().sum(axis=1)
    new = []
    for e in log["entries"]:
        tgt = e["target"]
        i = idx.get(tgt)
        if tgt in done or i is None or i < 1:
            continue
        # تغطية ناقصة (Yahoo لسه محدّثش كل الأسهم): نستنى بدل ما نحكم على نص السوق
        if cnt.iloc[i] < 0.7 * float(cnt.iloc[max(0, i - 21):i].median()):
            continue
        ret = (C.iloc[i] / C.iloc[i - 1] - 1)
        ret = ret[np.isfinite(ret) & (ret.abs() <= bt.MAX_ABS_RET)]
        labs = e.get("labels", {})
        nws = e.get("news", {})
        picks = [{"n": n, "r": float(ret[n + ".CA"] * 100), "lab": labs.get(n), "nw": nws.get(n)}
                 for n in e["picks"] if (n + ".CA") in ret.index]
        if len(picks) < max(3, len(e["picks"]) // 2) or len(ret) < 30:
            continue
        lp = [x for x in picks if x["lab"]]
        rec = {"target": tgt, "data_date": e["data_date"], "n": len(picks),
               "dir_n": len(lp),
               "dir_ok": sum(1 for x in lp if (x["r"] > 0) == (x["lab"] == dr.BUY) and x["r"] != 0),
               "up": sum(1 for x in picks if x["r"] > 0), "down": sum(1 for x in picks if x["r"] < 0),
               "avg": round(float(np.mean([x["r"] for x in picks])), 3),
               "mkt_n": int(len(ret)), "mkt_up": int((ret > 0).sum()),
               "mkt_avg": round(float(ret.mean() * 100), 3), "picks": picks}
        sc["days"].append(rec)
        new.append(rec)
    if new:
        sc["days"] = sorted(sc["days"], key=lambda d: d["target"])[-250:]
        _save_json(SCORE_FILE, sc)
    return new


def pick_line(x):
    """سطر سهم في رسالة النتيجة: لو كان ليه تصنيف شراء/بيع بنقول صح ولا غلط، وإلا طلع/نزل بس."""
    if x.get("lab"):
        ok = x["r"] != 0 and (x["r"] > 0) == (x["lab"] == dr.BUY)
        return f"{'✅' if ok else '❌'} {x['n']} {x['r']:+.1f}% | قلنا {x['lab']}"
    mark = "✅" if x["r"] > 0 else ("❌" if x["r"] < 0 else "➖")
    return f"{mark} {x['n']} {x['r']:+.1f}%"


def scorecard_message(rec):
    """رسالة تليجرام: كام سهم من ترشيحات الجلسة طلعوا فعلاً + الحصيلة التراكمية."""
    wd = datetime.strptime(rec["target"], "%Y-%m-%d").weekday()
    n, up = rec["n"], rec["up"]
    L_ = [f"📝 نتيجة ترشيحات جلسة {AR_DAYS[wd]} {rec['target']}",
          f"طلع {up} من {n} | متوسط التغير {rec['avg']:+.2f}%",
          f"السوق كله: طلع {rec['mkt_up'] / rec['mkt_n'] * 100:.0f}% من الأسهم | متوسط التغير {rec['mkt_avg']:+.2f}%",
          ""]
    for x in rec["picks"]:
        L_.append(pick_line(x))
    if rec.get("dir_n"):
        L_.insert(3, f"تصنيف شراء/بيع: صح في {rec['dir_ok']} من {rec['dir_n']}")
    days = _load_json(SCORE_FILE, {"days": []})["days"]
    if len(days) >= 2:
        tn, tu = sum(d["n"] for d in days), sum(d["up"] for d in days)
        mk = sum(d["mkt_up"] for d in days) / sum(d["mkt_n"] for d in days) * 100
        edge = sum(d["avg"] - d["mkt_avg"] for d in days) / len(days)
        L_ += ["", f"📈 الحصيلة بعد {len(days)} جلسة: الترشيحات طلعت {tu / tn * 100:.0f}% من المرات "
                   f"مقابل {mk:.0f}% للسوق | الفرق في المتوسط {edge:+.2f}% في اليوم"]
        dn, dok = sum(d.get("dir_n", 0) for d in days), sum(d.get("dir_ok", 0) for d in days)
        if dn:
            L_.append(f"🎯 تصنيف شراء/بيع: صح في {dok} من {dn} ({dok / dn * 100:.0f}%) من أول ما اتضاف")
        ns = nw.agreement_summary(days)
        if ns:
            L_.append(ns)
        if len(days) < 30:
            L_.append(f"⚠️ العينة صغيرة ({len(days)} جلسة × {n} سهم). النسبة بتتأرجح جدًا، "
                      f"متحكمش على الموديل قبل 30 جلسة على الأقل.")
    else:
        L_ += ["", "⚠️ دي أول جلسة بتتقيّم. الحصيلة التراكمية بتبدأ من الجلسة التانية."]
    return "\n".join(L_)


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
    ap.add_argument("--no-news", action="store_true", help="من غير أخبار الأسهم")
    ap.add_argument("--no-record", action="store_true", help="متسجلش الترشيحات في picks_log.json (للتجربة)")
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
        C, H, L, V, O, note = extend_with_state(C, H, L, V, O, now)
        # 1) نتيجة ترشيحات الجلسات اللي خلصت (قبل ما نسجّل ترشيحات الجلسة الجاية)
        scored = score_pending(C)
        text, rows, data_date, forecast = build_report(C, H, L, V, O, a, now, note)
        if not a.no_record:
            record_picks(rows, next_trading_day(now), data_date, now)
            _save_json(os.path.join(FORECAST_DIR, "d1.json"), forecast)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "report_latest.md"), "w", encoding="utf-8") as fh:
            fh.write(text)
        # أي error قديم يتشال بعد نجاح التشغيل
        err = os.path.join(OUT_DIR, "error_latest.txt")
        if os.path.exists(err):
            os.remove(err)
        print(text)
        if not a.no_telegram:
            for rec in scored[-3:]:
                bt.send_telegram(scorecard_message(rec))
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
