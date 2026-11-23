#!/usr/bin/env python3
"""
تحديث قايمة الأسهم + القطاعات + لقطة الأرقام الأساسية من TradingView.

1) أي سهم موجود على TradingView ومش في tickers.txt: لو Yahoo عنده تاريخ كفاية (≥ 120 جلسة) بيتضاف للقايمة
   تحت قسم "اتضاف أوتوماتيك". (Yahoo هو مصدر التاريخ اللي الموديل بيتدرب عليه.)
2) data/sectors.json: قطاع وصناعة كل سهم (بيتستخدم في مدخلات الموديل: أداء السهم مقارنة بقطاعه).
3) data/fundamentals/<تاريخ>.json: لقطة أسبوعية لـ P/E والقيمة السوقية والأرباح والتوزيعات. مش داخلة في الموديل
   لسه لأن مفيش تاريخ قديم ليها (استخدام أرقام النهاردة على الماضي = غش)، بنجمعها عشان تتختبر بعد كام شهر.
"""
import json
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
TICKERS = os.path.join(HERE, "tickers.txt")
SECTORS = os.path.join(HERE, "data", "sectors.json")
FUND_DIR = os.path.join(HERE, "data", "fundamentals")
URL = "https://scanner.tradingview.com/egypt/scan"
COLS = ["name", "description", "sector", "industry", "market_cap_basic", "price_earnings_ttm",
        "earnings_per_share_basic_ttm", "dividend_yield_recent", "price_book_fq", "close", "Value.Traded"]
MARK = "# --- اتضاف أوتوماتيك من TradingView (update_universe.py) ---"


def tv_all():
    body = {"columns": COLS, "range": [0, 1000], "markets": ["egypt"],
            "filter": [{"left": "type", "operation": "in_range", "right": ["stock", "dr", "fund"]}]}
    r = requests.post(URL, json=body, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    return [dict(zip(COLS, x["d"])) for x in r.json()["data"]]


def current():
    with open(TICKERS, encoding="utf-8") as fh:
        raw = "\n".join(l for l in fh if not l.lstrip().startswith("#"))
    return {t.upper() for t in re.findall(r"[A-Za-z0-9]{2,8}\.CA", raw)}


def main():
    rows = tv_all()
    print(f"[universe] TradingView: {len(rows)} سهم")
    os.makedirs(os.path.dirname(SECTORS), exist_ok=True)
    sec = {str(r["name"]).upper(): {"sector": r.get("sector") or "?", "industry": r.get("industry") or "?",
                                    "desc": r.get("description")} for r in rows}
    with open(SECTORS, "w", encoding="utf-8") as fh:
        json.dump(sec, fh, ensure_ascii=False, indent=0)
    today = datetime.now(ZoneInfo("Africa/Cairo")).strftime("%Y-%m-%d")
    os.makedirs(FUND_DIR, exist_ok=True)
    have = sorted(os.listdir(FUND_DIR))
    if not have or (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(have[-1][:10], "%Y-%m-%d")).days >= 7:
        with open(os.path.join(FUND_DIR, f"{today}.json"), "w", encoding="utf-8") as fh:
            json.dump({str(r["name"]).upper(): [r.get(c) for c in COLS[4:9]] for r in rows}, fh, separators=(",", ":"))
        print(f"[universe] لقطة أرقام أساسية {today}")
    cur = current()
    cand = sorted(f"{str(r['name']).upper()}.CA" for r in rows if f"{str(r['name']).upper()}.CA" not in cur)
    if "--no-add" in sys.argv or not cand:
        print(f"[universe] مفيش إضافة ({len(cand)} مرشح)")
        return
    import yfinance as yf
    added = []
    for t in cand:
        try:
            h = yf.Ticker(t).history(period="2y", interval="1d", auto_adjust=True)
            n = int(h["Close"].notna().sum()) if len(h) else 0
        except Exception as e:  # noqa: BLE001
            n = 0
            print(f"[universe] {t}: {str(e)[:80]}")
        print(f"[universe] {t}: {n} جلسة على Yahoo")
        if n >= 120:
            added.append(t)
    if added:
        with open(TICKERS, encoding="utf-8") as fh:
            txt = fh.read()
        if MARK not in txt:
            txt = txt.rstrip("\n") + "\n\n" + MARK + "\n"
        txt = txt.rstrip("\n") + "\n" + "\n".join(f"{t}  # {sec[t[:-3]]['desc']}" for t in added) + "\n"
        with open(TICKERS, "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(f"[universe] اتضاف {len(added)} سهم: {', '.join(added)}")


if __name__ == "__main__":
    main()
