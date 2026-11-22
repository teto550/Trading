"""
مصدر أسعار EGX الحي (بديل Mubasher).

Mubasher غيّر موقعه لتطبيق Angular من حوالي 12 نوفمبر 2026، فالسعر مبقاش موجود في HTML
والـAPI بتاعه محتاج توكن → كل الـruns من ساعتها كانت بتفشل وstate.json وقف عند 12/11.
البديل: TradingView scanner - طلب واحد بيرجّع كل أسهم EGX (افتتاح/أعلى/أقل/قفلة/حجم/قفلة امبارح
+ تاريخ الشمعة)، متأخر 15 دقيقة أثناء الجلسة، ونهائي بعد القفل.

كمان Yahoo بيتأخر عن EGX جلسة أو أكتر، فبنحفظ شمعة كل جلسة من TradingView في data/tv_bars.json
(بتتعمل commit مع التقارير)، وextend_panel بتكمّل بيها الأيام اللي Yahoo لسه منزّلهاش.
"""
import json
import os
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BARS_FILE = os.path.join(HERE, "data", "tv_bars.json")
URL = "https://scanner.tradingview.com/egypt/scan"
COLS = ["name", "open", "high", "low", "close", "volume", "close[1]", "time", "change", "Value.Traded", "description"]
TZ = ZoneInfo("Africa/Cairo")
_lock = threading.Lock()
_cache = {"t": 0.0, "data": None}


def tv_snapshot(max_age=60, retries=3):
    """{رمز: {open, high, low, close, volume, prev, date, chg, value, desc}} لكل أسهم EGX. كاش max_age ثانية."""
    with _lock:
        if _cache["data"] is not None and time.time() - _cache["t"] < max_age:
            return _cache["data"]
        body = {"columns": COLS, "range": [0, 1000], "markets": ["egypt"],
                "filter": [{"left": "type", "operation": "in_range", "right": ["stock", "dr", "fund"]}]}
        last = None
        for k in range(retries):
            try:
                r = requests.post(URL, json=body, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                r.raise_for_status()
                out = {}
                for row in r.json().get("data", []):
                    d = dict(zip(COLS, row["d"]))
                    if not d.get("close") or d["close"] <= 0:
                        continue
                    ts = d.get("time")
                    day = (datetime.fromtimestamp(ts, timezone.utc).astimezone(TZ).strftime("%Y-%m-%d") if ts else None)
                    out[str(d["name"]).upper()] = {
                        "open": d.get("open"), "high": d.get("high"), "low": d.get("low"), "close": float(d["close"]),
                        "volume": d.get("volume") or 0, "prev": d.get("close[1]"), "date": day,
                        "chg": d.get("change"), "value": d.get("Value.Traded"), "desc": d.get("description")}
                if len(out) < 50:
                    raise ValueError(f"TradingView رجّع {len(out)} سهم بس")
                _cache.update(t=time.time(), data=out)
                return out
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (k + 1))
        raise RuntimeError(f"TradingView scanner فشل: {last}")


def price(ticker):
    """آخر سعر للسهم (COMI أو COMI.CA)."""
    rec = tv_snapshot().get(ticker.upper().replace(".CA", ""))
    if not rec:
        raise ValueError(f"{ticker} مش موجود في TradingView")
    return rec["close"], rec


def session_closed(now=None):
    now = now or datetime.now(TZ)
    return now.weekday() not in (6, 0, 1, 2, 3) or (now.hour, now.minute) >= (15, 0) or now.hour < 10


# ============================================================
# أرشيف شمعات الجلسات (بيكمّل تأخير Yahoo)
# ============================================================
def load_bars():
    try:
        with open(BARS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def record_today(now=None):
    """بيحفظ شمعة آخر جلسة مقفولة في الأرشيف. بيرجّع تاريخها أو None."""
    now = now or datetime.now(TZ)
    snap = tv_snapshot()
    days = {}
    for s, r in snap.items():
        if r["date"]:
            days[r["date"]] = days.get(r["date"], 0) + 1
    if not days:
        return None
    day = max(days, key=days.get)
    if day == now.strftime("%Y-%m-%d") and not session_closed(now):
        return None                                   # الجلسة لسه شغالة: مش قفلة
    bars = load_bars()
    bars[day] = {s: [r["open"], r["high"], r["low"], r["close"], r["volume"], r["prev"]]
                 for s, r in snap.items() if r["date"] == day}
    keep = sorted(bars)[-120:]
    os.makedirs(os.path.dirname(BARS_FILE), exist_ok=True)
    with open(BARS_FILE, "w", encoding="utf-8") as fh:
        json.dump({k: bars[k] for k in keep}, fh, separators=(",", ":"))
    return day


def extend_panel(C, H, L, V, O, now=None, min_frac=0.6, live=True):
    """بيضيف للـpanel (أسعار Yahoo المعدّلة) الجلسات اللي Yahoo لسه منزّلهاش، من أرشيف TradingView
    (+ اللقطة الحالية لو live). السعر بيتربط بالتغير عن قفلة امبارح (prev) فمفيش مشكلة تعديل التوزيعات.
    بيرجّع (C, H, L, V, O, ملاحظة أو None)."""
    import numpy as np
    import pandas as pd
    now = now or datetime.now(TZ)
    bars = load_bars()
    if live:
        try:
            d = record_today(now)
            if d:
                bars = load_bars()
        except Exception as e:  # noqa: BLE001
            print(f"[egxdata] لقطة TradingView فشلت: {str(e)[:150]}")
    last = C.index[-1]
    added = []
    for day in sorted(bars):
        ts = pd.Timestamp(day)
        if ts <= last:
            continue
        if ts.date() == now.date() and not session_closed(now):
            continue
        b = bars[day]
        prevC = C.iloc[-1]
        new = {k: {} for k in "CHLVO"}
        for t in C.columns:
            rec = b.get(t.replace(".CA", ""))
            pc = prevC.get(t)
            if not rec or not np.isfinite(pc) or not rec[5] or rec[5] <= 0:
                continue
            f = pc / rec[5]                                   # معامل: قفلة Yahoo المعدّلة / قفلة امبارح عند TV
            if not (0.5 < f < 2.0):
                continue
            o, h, l, c, v = rec[0], rec[1], rec[2], rec[3], rec[4]
            new["C"][t] = c * f
            new["O"][t] = o * f if o else np.nan
            new["H"][t] = h * f if h else c * f
            new["L"][t] = l * f if l else c * f
            new["V"][t] = float(v or 0)
        if len(new["C"]) < min_frac * int(prevC.notna().sum()):
            break
        fs = np.array([new["C"][t] / prevC[t] / (b[t.replace(".CA", "")][3] / b[t.replace(".CA", "")][5])
                       for t in new["C"]])
        # لو قفلة امبارح عند TV مش هي آخر قفلة عند Yahoo (فيه جلسة ناقصة في النص) المعاملات بتتبعتر: نقف
        if np.median(np.abs(np.log(np.array([prevC[t] for t in new["C"]]) /
                                    np.array([b[t.replace(".CA", "")][5] for t in new["C"]])))) > 0.03:
            print(f"[egxdata] {day}: قفلة امبارح عند TradingView مش متطابقة مع آخر يوم عندنا، مش هضيفها")
            break
        del fs
        out = []
        for k, df in zip("CHLVO", (C, H, L, V, O)):
            row = pd.Series(new[k]).reindex(df.columns).to_frame(ts).T
            out.append(pd.concat([df, row]))
        C, H, L, V, O = out
        added.append(day)
    note = None
    if added:
        note = f"ℹ️ اتضاف من TradingView {len(added)} جلسة Yahoo لسه منزّلهاش ({', '.join(added)})."
    return C, H, L, V, O, note
