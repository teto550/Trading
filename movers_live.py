#!/usr/bin/env python3
"""
مراقبة لحظية لأسهم EGX أثناء الجلسة: بتبعت تليجرام لما سهم يطلع أو ينزل بسرعة/بنسبة كبيرة.
- بتدور في loop طول الجلسة (10:00 - 14:30 القاهرة) وتسحب كل الأسهم من Mubasher كل ~دقيقتين ونص.
- تنبيه "مستوى اليوم": تغيير عن قفلة امبارح عدّى +3/+5/+8% (أو -3/-5/-8%) - كل مستوى مرة في اليوم.
- تنبيه "حركة سريعة": تغيير 2% أو أكتر في آخر ~30 دقيقة.
- بيبعت رسالة واحدة مجمّعة في كل دورة، ومعاها حالة السوق ككل.
مش توصية ولا توقع - ده رصد لحركة حصلت بالفعل.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import monitor_once as mo  # noqa: E402

TZ = ZoneInfo("Africa/Cairo")
STATE = os.path.join(HERE, "movers_state.json")
LEVELS = (3.0, 5.0, 8.0)
FAST_PCT = 2.0
FAST_MIN = 30
FAST_COOLDOWN_MIN = 45
INTERVAL = int(os.environ.get("MOVERS_INTERVAL", "150"))
FORCE = os.environ.get("FORCE_RUN", "false").lower() == "true"
START = (9, 55)
END = (14, 32)
TOP = 8
PRED_TIMES = ((10, 30), (11, 30), (12, 30), (13, 30))   # مواعيد التوقع اللحظي (القاهرة)
SCORE_AT = (14, 25)
CONFIRM = 0.3   # تأكيد الحركة الحية للتوقع (% عن قفلة امبارح)
SCORECARD = os.path.join(HERE, "live_scorecard.json")


def now():
    return datetime.now(TZ)


def hm(d):
    return (d.hour, d.minute)


def trading_day(d):
    return d.weekday() in (6, 0, 1, 2, 3)


def load_state(today):
    try:
        with open(STATE, encoding="utf-8") as f:
            s = json.load(f)
        if s.get("date") == today:
            return s
    except Exception:  # noqa: BLE001
        pass
    return {"date": today, "levels": {}, "fast": {}, "preds": [], "pred_done": [], "scored": False}


def save_state(s):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False)


def prev_closes(today):
    """قفلة آخر يوم قبل النهارده من state.json بتاع المونيتور."""
    out = {}
    try:
        with open(os.path.join(HERE, "state.json"), encoding="utf-8") as f:
            st = json.load(f)["stocks"]
    except Exception:  # noqa: BLE001
        return out
    for t, v in st.items():
        d, p = v.get("dates") or [], v.get("daily") or []
        if not p:
            continue
        out[t] = p[-2] if (d and d[-1] >= today and len(p) >= 2) else p[-1]
    return out


def load_labels(today):
    """تصنيف الموديل (شراء/بيع) للجلسة دي من forecast/d1.json لو الـtarget هو النهارده."""
    try:
        with open(os.path.join(HERE, "forecast", "d1.json"), encoding="utf-8") as f:
            d = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    if d.get("target") != today and not FORCE:
        print(f"[توقع] forecast/d1 للجلسة {d.get('target')} مش {today} - مفيش توقع لحظي")
        return {}
    return {k + ".CA": v.get("label") for k, v in d["stocks"].items()}


def predict(labels, chg, ok, med, state, t_now):
    """توقع لحظي = تصنيف الموديل + تأكيد من الحركة الحية، ومش بنتوقع ضد السوق لو بينهار/بيطير."""
    ups, downs = [], []
    done = {(p["t"], p["side"]) for p in state["preds"]}
    for t, c in chg.items():
        lab = labels.get(t)
        if lab == "شراء" and c >= CONFIRM and med > -1.5 and (t, "up") not in done:
            ups.append((c, t))
        elif lab == "بيع" and c <= -CONFIRM and med < 1.5 and (t, "down") not in done:
            downs.append((c, t))
    ups.sort(reverse=True)
    downs.sort()
    for side, rows in (("up", ups[:TOP]), ("down", downs[:TOP])):
        for c, t in rows:
            state["preds"].append({"t": t, "side": side, "price": ok[t], "chg": c, "med": med,
                                   "time": f"{t_now:%H:%M}"})
    return ups[:TOP], downs[:TOP]


def pred_message(t_now, med, ups, downs, ok):
    lines = [f"🔮 توقع لحظي - {t_now:%H:%M}" + (" (تجربة)" if FORCE else ""),
             f"السوق ككل: {med:+.1f}% | التوقع = تصنيف الموديل لجلسة النهارده + تأكيد من الحركة الحية", ""]
    if ups:
        lines.append("🟢 مرشحة تكمل طلوع لقفلة النهارده:")
        lines += [f"• {t.replace('.CA', '')}: {ok[t]:.2f} ({c:+.1f}%)" for c, t in ups]
    if downs:
        lines += ["", "🔴 مرشحة تكمل نزول لقفلة النهارده:"]
        lines += [f"• {t.replace('.CA', '')}: {ok[t]:.2f} ({c:+.1f}%)" for c, t in downs]
    lines += ["", "⚠️ ده توقع تجريبي: الباكتست ماأثبتش ميزة للجلسة القصيرة بعد التكلفة، وأنا بسجّل كل توقع "
              "وهقيّمه عند القفل وأبعتلك نسبة نجاحه الحقيقية. متعتمدش عليه لوحده."]
    return "\n".join(lines)


def score_preds(state, chg, ok, med, t_now):
    P = state["preds"]
    rows = []
    for p in P:
        t = p["t"]
        if t not in ok:
            continue
        mv = (ok[t] / p["price"] - 1) * 100
        mk = med - p["med"]
        sgn = 1 if p["side"] == "up" else -1
        rows.append({"t": t, "side": p["side"], "move": mv, "excess": sgn * (mv - mk), "dir_ok": sgn * mv > 0})
    if not rows:
        return None
    try:
        with open(SCORECARD, encoding="utf-8") as f:
            sc = json.load(f)
    except Exception:  # noqa: BLE001
        sc = []
    n = len(rows)
    rec = {"date": state["date"], "n": n, "hit": sum(r["dir_ok"] for r in rows) / n * 100,
           "excess": sum(r["excess"] for r in rows) / n,
           "beat": sum(r["excess"] > 0 for r in rows) / n * 100}
    sc = [x for x in sc if x["date"] != rec["date"]] + [rec]
    if not FORCE:
        with open(SCORECARD, "w", encoding="utf-8") as f:
            json.dump(sc, f, ensure_ascii=False)
    tn = sum(x["n"] for x in sc)
    cum_h = sum(x["hit"] * x["n"] for x in sc) / tn
    cum_e = sum(x["excess"] * x["n"] for x in sc) / tn
    best = sorted(rows, key=lambda r: -r["excess"])[:3]
    worst = sorted(rows, key=lambda r: r["excess"])[:3]
    L = [f"🏁 نتيجة التوقع اللحظي - {state['date']}" + (" (تجربة)" if FORCE else ""),
         f"{n} توقع: اتجاه صح في {rec['hit']:.0f}% | متوسط الزيادة عن السوق (من وقت التوقع للقفل): {rec['excess']:+.2f}% | غلبوا السوق: {rec['beat']:.0f}%",
         f"التراكمي ({len(sc)} يوم، {tn} توقع): اتجاه صح {cum_h:.0f}% | زيادة عن السوق {cum_e:+.2f}% للتوقع",
         "أحسن: " + "، ".join(f"{r['t'].replace('.CA','')} {r['excess']:+.1f}%" for r in best),
         "أسوأ: " + "، ".join(f"{r['t'].replace('.CA','')} {r['excess']:+.1f}%" for r in worst),
         "(الزيادة قبل تكلفة ~0.4% للصفقة، وغير مفيدة لو أقل منها)"]
    return "\n".join(L)


def fetch(t):
    try:
        return t, mo.fetch_price_mubasher(t)
    except Exception:  # noqa: BLE001
        return t, None


def cycle(tickers, prev, hist, state, t_now):
    with ThreadPoolExecutor(max_workers=10) as ex:
        res = dict(ex.map(fetch, tickers))
    ok = {t: p for t, p in res.items() if p}
    chg = {}
    for t, p in ok.items():
        pc = prev.get(t)
        if pc and pc > 0 and abs(p / pc - 1) <= 0.21:  # حد EGX 10%، أكتر من كده غالبًا بيانات غلط/تقسيم
            chg[t] = (p / pc - 1) * 100
    print(f"[{t_now:%H:%M}] أسعار: {len(ok)}/{len(tickers)} | بتغيير صالح: {len(chg)}")
    if len(chg) < 20:
        return None
    med = sorted(chg.values())[len(chg) // 2]
    ups, downs = [], []
    for t, c in chg.items():
        h = hist.setdefault(t, [])
        h.append((t_now.timestamp(), ok[t]))
        del h[:-60]
        # 1) مستوى اليوم
        lv = state["levels"].setdefault(t, {"up": 0, "down": 0})
        for L in LEVELS:
            if c >= L > lv["up"]:
                lv["up"] = L
                ups.append((c, t, f"عدّى +{L:.0f}% النهارده"))
            if c <= -L and L > lv["down"]:
                lv["down"] = L
                downs.append((c, t, f"عدّى -{L:.0f}% النهارده"))
        # 2) حركة سريعة آخر FAST_MIN دقيقة
        ref = next((p for ts, p in h if t_now.timestamp() - ts <= FAST_MIN * 60 + 60), None)
        if ref and ref > 0:
            m = (ok[t] / ref - 1) * 100
            for side, hit in (("up", m >= FAST_PCT), ("down", m <= -FAST_PCT)):
                last = state["fast"].get(f"{t}|{side}", 0)
                if hit and t_now.timestamp() - last > FAST_COOLDOWN_MIN * 60:
                    state["fast"][f"{t}|{side}"] = t_now.timestamp()
                    (ups if side == "up" else downs).append((c, t, f"{m:+.1f}% في آخر {FAST_MIN} دقيقة"))
    return med, ups, downs, chg, ok


def fmt(rows, icon, ok):
    seen, out = set(), []
    for c, t, why in sorted(rows, key=lambda r: -abs(r[0])):
        if t in seen:
            continue
        seen.add(t)
        out.append(f"{icon} {t.replace('.CA', '')}: {ok[t]:.2f} ({c:+.1f}% عن امبارح) - {why}")
    return out[:TOP], max(0, len(seen) - TOP)


def message(t_now, med, ups, downs, ok):
    lines = [f"📡 حركة لحظية - {t_now:%H:%M}" + (" (تجربة)" if FORCE else "")]
    lines.append(f"السوق ككل (الوسيط): {med:+.1f}% عن قفلة امبارح")
    if abs(med) >= 2:
        lines.append("⚠️ السوق كله بيتحرك، فحركة الأسهم دي ممكن تكون تبع السوق مش ليها خبر خاص.")
    for rows, icon, title in ((ups, "🟢", "طالع"), (downs, "🔴", "نازل")):
        body, extra = fmt(rows, icon, ok)
        if body:
            lines += ["", f"{title}:"] + body + ([f"... و{extra} تانيين"] if extra else [])
    lines += ["", "ده رصد لحركة حصلت فعلًا، مش توقع إنها هتكمل ولا توصية شراء أو بيع. الحركة السريعة ممكن ترجع."]
    return "\n".join(lines)


def main():
    t0 = now()
    today = f"{t0:%Y-%m-%d}"
    if not FORCE and not trading_day(t0):
        print("مش يوم تداول")
        return
    if not FORCE and hm(t0) > END:
        print("الجلسة خلصت")
        return
    tickers = mo.load_tickers()
    prev = prev_closes(today)
    state = load_state(today)
    state.setdefault("preds", [])
    state.setdefault("pred_done", [])
    labels = load_labels(today)
    hist = {}
    cycles = 0
    try:
        while True:
            t_now = now()
            if not FORCE:
                if hm(t_now) < START:
                    wait = ((START[0] - t_now.hour) * 60 + START[1] - t_now.minute) * 60
                    if wait > 45 * 60:
                        print("بدري جدًا، خروج")
                        return
                    time.sleep(min(wait, 60))
                    continue
                if hm(t_now) > END:
                    break
            r = cycle(tickers, prev, hist, state, t_now)
            cycles += 1
            if r:
                med, ups, downs, chg, ok = r
                if (ups or downs) and hm(t_now) >= (10, 0) or FORCE:
                    # أول دورة بنسجّل المستويات الحالية بس بدون إغراق لو كتير
                    if ups or downs:
                        mo.send_telegram(message(t_now, med, ups, downs, ok))
            if r and labels:
                med, ups, downs, chg, ok = r
                due = [pt for pt in PRED_TIMES if hm(t_now) >= pt and list(pt) not in state["pred_done"]]
                if FORCE and cycles == 1:
                    due = [PRED_TIMES[0]]
                if due:
                    state["pred_done"] += [list(pt) for pt in due]
                    pu, pd_ = predict(labels, chg, ok, med, state, t_now)
                    if pu or pd_:
                        mo.send_telegram(pred_message(t_now, med, pu, pd_, ok))
                if (hm(t_now) >= SCORE_AT or FORCE and cycles == 2) and not state["scored"]:
                    msg = score_preds(state, chg, ok, med, t_now)
                    state["scored"] = True
                    if msg:
                        mo.send_telegram(msg)
            save_state(state)
            if FORCE and cycles >= 2:
                break
            time.sleep(INTERVAL)
    finally:
        save_state(state)
        print(f"[خلص] دورات: {cycles}")


if __name__ == "__main__":
    main()
