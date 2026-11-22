"""Probe TradingView scanner fields (debug only)."""
import json, os, requests
OUT = "probe_out"; os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}
res = {}
for col in ["time", "update_time", "last_bar_update_time", "lp_time", "close_time", "market_status", "current_session",
            "change_from_open", "gap", "Perf.W", "average_volume_10d_calc", "relative_volume_10d_calc", "close[1]", "currency"]:
    body = {"columns": ["name", "close", col], "range": [0, 3], "markets": ["egypt"],
            "filter": [{"left": "name", "operation": "in_range", "right": ["COMI", "SWDY", "EFIH"]}]}
    r = requests.post("https://scanner.tradingview.com/egypt/scan", json=body, headers=UA, timeout=20)
    res[col] = [r.status_code, r.text[:400]]
open(os.path.join(OUT, "tv_fields.json"), "w").write(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
