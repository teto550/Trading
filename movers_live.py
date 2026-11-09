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
    return {"date": today, "levels": {}, "fast": {}}


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
            save_state(state)
            if FORCE and cycles >= 2:
                break
            time.sleep(INTERVAL)
    finally:
        save_state(state)
        print(f"[خلص] دورات: {cycles}")


if __name__ == "__main__":
    main()
