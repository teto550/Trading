#!/usr/bin/env python3
"""
================================================================
   EGX - توقع 5 و10 أيام (رسالة مستقلة عن تقرير جلسة بكره)
================================================================
كل يوم تداول بعد القفل:
  1) بيدرّب XGBoost (نفس موديل الباكتست متعدد الأيام) مرتين: هدف 5 أيام وهدف 10 أيام،
     وبيطلّع أقوى 10 أسهم ترتيبًا لكل فترة (ترتيب نسبي بين الأسهم، مش نسبة احتمال).
  2) بيسجّل الاختيارات في horizon/picks_log.json.
  3) بعد ما تعدّي 5 (أو 10) جلسات على أي توقع، بيحسب كام سهم من العشرة طلعوا فعلاً
     (من قفلة يوم التوقع لقفلة اليوم الـh) مقارنة بالسوق كله، وبيبعت النتيجة في رسالة لوحدها،
     مع حصيلة تراكمية لكل فترة (horizon/scorecard.json).

⚠️ الباكتست (2022+) بيقول إن 5 و10 أيام بعد التكلفة لسه مش مؤكدة (حوالي -2.2 و-0.6 bps/يوم)،
فالرسالة دي ترتيب إحصائي للمتابعة والقياس، مش توصية ولا توقع إن السهم هيطلع.

الاستخدام:
  python backtest/horizon_report.py                 # بيبعت تليجرام لو التوكن متحدد
  python backtest/horizon_report.py --no-telegram --no-record --source synthetic   # تجربة
================================================================
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import backtest as bt  # noqa: E402
import backtest_horizons as bh  # noqa: E402
import macro  # noqa: E402
import tomorrow_report as tr  # noqa: E402

OUT_DIR = os.path.join(ROOT, "horizon")
LOG_FILE = os.path.join(OUT_DIR, "picks_log.json")
SCORE_FILE = os.path.join(OUT_DIR, "scorecard.json")
HORIZONS = (5, 10)
TOP = 10
MAX_HORIZON_RET = 1.0      # أي تغير أكبر من 100% في الفترة غالبًا تقسيم/بيانات غلط


def _load(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return default


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


# ============================================================
# المدخلات والتدريب
# ============================================================
def prepare(C, H, L, V, O, min_turnover, M=None):
    f, mkt, _n, _noc, lock, liquid, _flat = bt.build_features(C, H, L, V, O, min_turnover)
    lr = np.log1p(f["r1"])
    f["r60"] = np.expm1(lr.rolling(60, min_periods=60).sum())
    f["r120"] = np.expm1(lr.rolling(120, min_periods=120).sum())
    Xcs, names = macro.build_X(f, mkt, liquid, C.shape[1], bh.XF2, bt.MF, M, bt.csrank)
    feat_ok = np.isfinite(Xcs).all(axis=2) & liquid.values & ~lock.values
    return {"C": C, "H": H, "L": L, "O": O, "Xcs": Xcs, "feat_ok": feat_ok, "names": names}


def top_picks(P, h, trees):
    """أقوى TOP أسهم لفترة h يوم، مدرّب على الصفوف اللي نتيجتها خلصت بس (مفيش تسريب من المستقبل)."""
    import xgboost as xgb
    C, O = P["C"], P["O"]
    tgt = bh.build_target(C, O, h)
    E = P["feat_ok"] & np.isfinite(tgt.values)
    if int(E.sum()) < 5000:
        raise bt.BacktestError(f"بيانات التدريب قليلة لفترة {h} أيام ({int(E.sum())} صف)")
    Y = (pd.DataFrame(np.where(E, tgt.values, np.nan)).rank(axis=1, pct=True) - 0.5).values
    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
        min_child_weight=50, reg_lambda=10.0, tree_method="hist", n_jobs=2, random_state=0)
    model.fit(P["Xcs"][E], Y[E].astype(np.float32))
    t = C.shape[0] - 1
    cols = np.where(P["feat_ok"][t])[0]
    if len(cols) < 40:
        raise bt.BacktestError(f"عدد الأسهم الصالحة النهاردة قليل ({len(cols)})")
    score = model.predict(P["Xcs"][t, cols])
    out = []
    for rank, i in enumerate(np.argsort(-score)[:TOP], 1):
        col = int(cols[i])
        lv = tr.levels(C, P["H"], P["L"], col, t)
        out.append({"rank": rank, "name": C.columns[col].replace(".CA", ""),
                    "last": float(C.iloc[t, col]), "lv": lv})
    return out


# ============================================================
# التسجيل والتقييم
# ============================================================
def record(date, picks_by_h, now, variant="base"):
    log = _load(LOG_FILE, {"entries": []})
    scored = {(d["date"], d["h"]) for d in _load(SCORE_FILE, {"days": []})["days"]}
    ent = log["entries"]
    for h, picks in picks_by_h.items():
        if (date, h) in scored:
            continue
        ent = [e for e in ent if not (e["date"] == date and e["h"] == h)]
        ent.append({"date": date, "h": h, "variant": variant, "picks": [p["name"] for p in picks],
                    "made": now.strftime("%Y-%m-%d %H:%M")})
    log["entries"] = sorted(ent, key=lambda e: (e["date"], e["h"]))[-400:]
    _save(LOG_FILE, log)


def score_pending(C):
    """كل توقع عدّى عليه h جلسات: التغير من قفلة يوم التوقع لقفلة الجلسة الـh، مقارنة بالسوق كله."""
    log = _load(LOG_FILE, {"entries": []})
    sc = _load(SCORE_FILE, {"days": []})
    done = {(d["date"], d["h"]) for d in sc["days"]}
    idx = {str(d.date()): i for i, d in enumerate(C.index)}
    cnt = C.notna().sum(axis=1)
    new = []
    for e in log["entries"]:
        i0 = idx.get(e["date"])
        if (e["date"], e["h"]) in done or i0 is None:
            continue
        i1 = i0 + e["h"]
        if i1 >= len(C):
            continue                                    # لسه ماخلصتش
        if cnt.iloc[i1] < 0.7 * float(cnt.iloc[max(0, i1 - 21):i1].median()):
            continue                                    # تغطية ناقصة، نستنى
        ret = C.iloc[i1] / C.iloc[i0] - 1
        ret = ret[np.isfinite(ret) & (ret.abs() <= MAX_HORIZON_RET)]
        picks = [{"n": n, "r": float(ret[n + ".CA"] * 100)} for n in e["picks"] if (n + ".CA") in ret.index]
        if len(picks) < max(3, len(e["picks"]) // 2) or len(ret) < 30:
            continue
        rec = {"date": e["date"], "h": e["h"], "variant": e.get("variant", "base"),
               "end": str(C.index[i1].date()), "n": len(picks),
               "up": sum(1 for x in picks if x["r"] > 0),
               "avg": round(float(np.mean([x["r"] for x in picks])), 3),
               "mkt_n": int(len(ret)), "mkt_up": int((ret > 0).sum()),
               "mkt_avg": round(float(ret.mean() * 100), 3), "picks": picks}
        sc["days"].append(rec)
        new.append(rec)
    if new:
        sc["days"] = sorted(sc["days"], key=lambda d: (d["date"], d["h"]))[-400:]
        _save(SCORE_FILE, sc)
    return new


def result_message(rec):
    h, n = rec["h"], rec["n"]
    L_ = [f"📝 نتيجة توقع {h} أيام اللي اتقال يوم {rec['date']} (لحد قفلة {rec['end']})",
          f"طلع {rec['up']} من {n} | متوسط التغير {rec['avg']:+.2f}%",
          f"السوق كله: طلع {rec['mkt_up'] / rec['mkt_n'] * 100:.0f}% من الأسهم | متوسط التغير {rec['mkt_avg']:+.2f}%",
          ""]
    for x in rec["picks"]:
        mark = "✅" if x["r"] > 0 else ("❌" if x["r"] < 0 else "➖")
        L_.append(f"{mark} {x['n']} {x['r']:+.1f}%")
    days = [d for d in _load(SCORE_FILE, {"days": []})["days"]
            if d["h"] == h and d.get("variant", "base") == rec.get("variant", "base")]
    if len(days) >= 2:
        tn, tu = sum(d["n"] for d in days), sum(d["up"] for d in days)
        mk = sum(d["mkt_up"] for d in days) / sum(d["mkt_n"] for d in days) * 100
        edge = sum(d["avg"] - d["mkt_avg"] for d in days) / len(days)
        L_ += ["", f"📈 الحصيلة بعد {len(days)} توقع ({h} أيام): طلعت {tu / tn * 100:.0f}% من المرات "
                   f"مقابل {mk:.0f}% للسوق | الفرق في المتوسط {edge:+.2f}% للفترة"]
    L_ += ["", "⚠️ التوقعات المتجاورة بتتشارك أيام، فمتتحسبش كعينات مستقلة. متحكمش على الموديل قبل شهور."]
    return "\n".join(L_)


# ============================================================
# رسالة التوقع الجديد
# ============================================================
def picks_message(date, picks_by_h, ms, names=None):
    L_ = [f"🔭 توقع 5 و10 أيام - مبني على قفلة {date}"]
    if names is not None:
        has = [n for n in names if n in macro.MACRO_MARKET or n == "beta60"]
        L_.append("المدخلات: " + ("الأسعار والسيولة + مؤشر EGX30 والدولار (جديدة، لسه ما اتقاسش أثرها في الباكتست)"
                                  if has else "الأسعار والسيولة بس (Yahoo مرجّعش EGX30 والدولار النهاردة)"))
    if ms:
        L_.append(f"{ms['tag']}: آخر جلسة {ms['mean1']:+.1f}% | {ms['breadth']:.0f}% من الأسهم طلعت | "
                  f"آخر 20 جلسة {ms['med20']:+.1f}% (وسيط الأسهم)")
        if ms["weak"]:
            L_.append("⚠️ السوق نازل بقوة: الترشيحات دي أضعف من العادي، والأحسن تتابع وتستنى.")
    for h, picks in picks_by_h.items():
        L_ += ["", f"🗓 أقوى {len(picks)} سهم لمدة {h} أيام:"]
        for p in picks:
            lv = p["lv"]
            line = f"{p['rank']}. {p['name']} | قفلة {tr.fmt_px(p['last'])}"
            if lv:
                line += (f" | RSI {lv['rsi']:.0f} | تذبذب يومي {lv['atr_pct']:.1f}%"
                         f" | دعم {tr.fmt_px(lv['support'])} | مقاومة {tr.fmt_px(lv['resist'])}")
            L_.append(line)
    L_ += ["",
           "هبعتلك بعد 5 و10 جلسات كام سهم منهم طلع فعلاً (من قفلة النهاردة).",
           "",
           "⚠️ اقرأ ده قبل ما تعتمد على الرسالة:",
           "• الترتيب نسبي بين الأسهم، مش نسبة احتمال ولا توقع إن السهم هيطلع.",
           "• الباكتست (2022+) بعد التكلفة: 5 أيام حوالي -2.2 bps/يوم و10 أيام حوالي -0.6، يعني الميزة لسه مش مؤكدة.",
           "• مش توصية شراء ولا بيع. القرار والمسؤولية ليك."]
    return "\n".join(L_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-turnover", type=float, default=1_000_000)
    ap.add_argument("--fresh", action="store_true", help="تجاهل كاش الأسعار وجيب بيانات جديدة")
    ap.add_argument("--source", choices=["yfinance", "synthetic"], default="yfinance")
    ap.add_argument("--no-macro", action="store_true", help="من غير EGX30 والدولار (للمقارنة)")
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="متسجلش في horizon/ (للتجربة)")
    a = ap.parse_args()
    now = datetime.now(ZoneInfo("Africa/Cairo"))
    try:
        if a.source == "synthetic":
            panel = bh.make_synthetic_h(True, T=900, N=120)
        else:
            if a.fresh:
                cache = os.path.join(bt.DATA_DIR, "prices.pkl")
                if os.path.exists(cache):
                    os.remove(cache)
            panel = bt.fetch_yfinance(bt.load_tickers())
        C, H, L, V, O = bt.clean_panel(panel)
        # قبل القفل أي صف بتاريخ النهاردة شمعة ناقصة، فبنشيله
        if now.weekday() in tr.TRADING_DAYS and (now.hour, now.minute) < tr.SESSION_CLOSE:
            keep = C.index.date != now.date()
            C, H, L, V, O = (x[keep] for x in (C, H, L, V, O))
        if a.source == "yfinance":
            C, H, L, V, O, _note = tr.extend_with_state(C, H, L, V, O, now)
        date = str(C.index[-1].date())
        M = None
        if not a.no_macro:
            try:
                if a.source == "synthetic":
                    M = macro.synthetic_macro((C / C.shift(1) - 1).mean(axis=1))
                else:
                    M = macro.fetch_macro(fresh=a.fresh)
            except Exception as e:  # noqa: BLE001
                print(f"[macro] فشل الجلب، هكمّل من غيره: {e}")
        P = prepare(C, H, L, V, O, a.min_turnover, M)
        variant = "macro" if any(n in macro.MACRO_MARKET or n == "beta60" for n in P["names"]) else "base"
        picks_by_h = {h: top_picks(P, h, 150) for h in HORIZONS}
        ms = tr.market_status(C, C.shape[0] - 1)
        scored = []
        if not a.no_record:
            scored = score_pending(C)
            record(date, picks_by_h, now, variant)
        text = picks_message(date, picks_by_h, ms, P["names"])
        os.makedirs(OUT_DIR, exist_ok=True)
        if not a.no_record:
            with open(os.path.join(OUT_DIR, "report_latest.md"), "w", encoding="utf-8") as fh:
                fh.write(text)
            err = os.path.join(OUT_DIR, "error_latest.txt")
            if os.path.exists(err):
                os.remove(err)
        print(text)
        for rec in scored:
            print("\n" + result_message(rec))
        if not a.no_telegram:
            for rec in scored[-4:]:
                bt.send_telegram(result_message(rec))
            tr.send_long(text)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "error_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M}\n{tb}")
        if not a.no_telegram:
            kind = "فشل (متوقع)" if isinstance(e, bt.BacktestError) else "خطأ في الكود"
            bt.send_telegram(f"🔭❌ توقع 5 و10 أيام وقع - {kind}:\n{type(e).__name__}: {str(e)[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
