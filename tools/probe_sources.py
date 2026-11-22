"""Probe live data sources from a GitHub runner and save raw outputs (debug only)."""
import json, os, re, traceback
import requests
OUT = "probe_out"; os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
def save(name, txt): open(os.path.join(OUT, name), "w", encoding="utf-8").write(txt)
def tryit(name, fn):
    try: fn()
    except Exception: save(name + ".err", traceback.format_exc())
def mub():
    for host in ("english.mubasher.info", "www.mubasher.info"):
        r = requests.get(f"https://{host}/markets/EGX/stocks/COMI", headers=UA, timeout=20)
        save(f"mub_{host}.html", f"status {r.status_code}\n" + r.text)
def mub_api():
    for u in ["https://english.mubasher.info/api/1/stocks/prices/all?country=eg",
              "https://www.mubasher.info/api/1/stocks/prices/all?country=eg",
              "https://english.mubasher.info/api/1/listed-companies?country=eg&size=500",
              "https://english.mubasher.info/api/1/stocks/COMI/chart?market=EGX"]:
        try:
            r = requests.get(u, headers=UA, timeout=20); save("mubapi_" + re.sub(r"\W", "_", u[8:])[:80] + ".txt", f"{r.status_code}\n{r.text[:20000]}")
        except Exception as e: save("mubapi_err_" + re.sub(r"\W", "_", u[8:])[:60] + ".txt", str(e))
def tv():
    body = {"filter": [{"left": "type", "operation": "in_range", "right": ["stock", "dr", "fund"]}],
            "columns": ["name", "description", "close", "open", "high", "low", "volume", "change", "update_mode", "exchange", "Value.Traded"],
            "sort": {"sortBy": "name", "sortOrder": "asc"}, "range": [0, 600], "markets": ["egypt"]}
    r = requests.post("https://scanner.tradingview.com/egypt/scan", json=body, headers=UA, timeout=20)
    save("tv_scan.json", f"{r.status_code}\n{r.text[:200000]}")
def tvhist():
    import time
    r = requests.get("https://scanner.tradingview.com/symbol", params={"symbol": "EGX:COMI", "fields": "close,open,high,low,volume,change"}, headers=UA, timeout=20)
    save("tv_symbol.txt", f"{r.status_code}\n{r.text[:5000]}")
def yf_():
    import yfinance as yf
    h = yf.Ticker("COMI.CA").history(period="15d", interval="1d")
    save("yf_comi.txt", h.to_string())
    h = yf.Ticker("COMI.CA").history(period="2d", interval="5m")
    save("yf_comi_5m.txt", h.tail(20).to_string())
for n, f in [("mub", mub), ("mubapi", mub_api), ("tv", tv), ("tvsym", tvhist), ("yf", yf_)]:
    tryit(n, f)
print(os.listdir(OUT))
