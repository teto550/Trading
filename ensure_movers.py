#!/usr/bin/env python3
"""بيضمن إن المراقبة اللحظية شغالة وقت الجلسة: GitHub cron بيسقّط تشغيلها، فالـjob بتاع تليجرام (الشغال طول اليوم تقريبًا)
بيشغّل ده في الخلفية، وهو بيطلب تشغيل movers.yml لو الجلسة مفتوحة ومفيش run شغال."""
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

REPO = os.environ.get("GITHUB_REPOSITORY", "teto550/Trading")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
API = f"https://api.github.com/repos/{REPO}/actions/workflows/movers.yml"
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}


def in_session(d):
    return d.weekday() in (6, 0, 1, 2, 3) and (9, 50) <= (d.hour, d.minute) <= (14, 15)


def active():
    for st in ("in_progress", "queued"):
        r = requests.get(f"{API}/runs", headers=H, params={"status": st, "per_page": 5}, timeout=20)
        r.raise_for_status()
        if r.json().get("total_count", 0) > 0:
            return True
    return False


def main():
    while True:
        try:
            d = datetime.now(ZoneInfo("Africa/Cairo"))
            if TOKEN and in_session(d) and not active():
                r = requests.post(f"{API}/dispatches", headers=H, json={"ref": "main"}, timeout=20)
                print(f"[ensure_movers] {d:%H:%M} dispatch -> {r.status_code}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ensure_movers] {str(e)[:120]}", flush=True)
        time.sleep(300)


if __name__ == "__main__":
    main()
