#!/usr/bin/env python3
"""نبض المواعيد: GitHub cron بيتأخر ساعات وبيسقّط تشغيلات (مثلاً المراقبة اللحظية كانت بتشتغل مرة واحدة في اليوم
بعد القفل، وتقرير المسا كان بيتأخر لـ12 بالليل). الـjob بتاع تليجرام (commands.yml، شغال ~6 ساعات) بيشغّل ده في الخلفية،
وهو بيطلب تشغيل كل workflow في ميعاده لو مفيش run شغال أو اتعمل قريب:
  - movers.yml   : طول الجلسة (لو مش شغال)
  - monitor.yml  : كل ~30 دقيقة في الجلسة + مرة بعد القفل
  - tomorrow.yml : مرة بعد القفل (فيه إشارات الدخول/الخروج 🧭 + تقرير الجلسة الجاية)
  - horizon.yml / paper_trade.yml : مرة بعد القفل
وقبل ما الـjob بتاعه يخلص، بيطلب commands.yml تاني عشان النبض ميقفش."""
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

REPO = os.environ.get("GITHUB_REPOSITORY", "teto550/Trading")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
REF = os.environ.get("GITHUB_REF_NAME", "main") or "main"
API = f"https://api.github.com/repos/{REPO}/actions/workflows"
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}
TZ = ZoneInfo("Africa/Cairo")
JOB_MINUTES = int(os.environ.get("JOB_MINUTES", "333"))
T0 = time.time()


def trading(d):
    return d.weekday() in (6, 0, 1, 2, 3)


def hm(d):
    return (d.hour, d.minute)


# (الملف, هل وقته دلوقتي؟, أقل مدة بين تشغيلين بالدقايق أو "day" = مرة بعد القفل كل يوم)
JOBS = [
    ("movers.yml", lambda d: trading(d) and (9, 50) <= hm(d) <= (14, 15), 0),
    ("monitor.yml", lambda d: trading(d) and (10, 5) <= hm(d) <= (15, 30), 28),
    ("tomorrow.yml", lambda d: trading(d) and (15, 5) <= hm(d) <= (23, 30), "day"),
    ("horizon.yml", lambda d: trading(d) and (15, 40) <= hm(d) <= (23, 30), "day"),
    ("paper_trade.yml", lambda d: trading(d) and (15, 50) <= hm(d) <= (23, 30), "day"),
]
AFTER_CLOSE = (14, 50)


def runs(wf, **params):
    r = requests.get(f"{API}/{wf}/runs", headers=H, params={"per_page": 5, **params}, timeout=20)
    r.raise_for_status()
    return r.json().get("workflow_runs", [])


def active(wf):
    return any(runs(wf, status=st) for st in ("in_progress", "queued"))


def last_created(wf):
    rs = runs(wf)
    if not rs:
        return None
    return datetime.strptime(rs[0]["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(TZ)


def dispatch(wf, d):
    r = requests.post(f"{API}/{wf}/dispatches", headers=H, json={"ref": REF}, timeout=20)
    print(f"[ensure] {d:%H:%M} dispatch {wf} -> {r.status_code}", flush=True)


def tick(d):
    for wf, when, gap in JOBS:
        try:
            if not when(d) or active(wf):
                continue
            if gap == 0:
                dispatch(wf, d)
                continue
            last = last_created(wf)
            if gap == "day":
                close_today = d.replace(hour=AFTER_CLOSE[0], minute=AFTER_CLOSE[1], second=0, microsecond=0)
                if last is None or last < close_today:
                    dispatch(wf, d)
            elif last is None or d - last >= timedelta(minutes=gap):
                dispatch(wf, d)
        except Exception as e:  # noqa: BLE001
            print(f"[ensure] {wf}: {str(e)[:150]}", flush=True)


def main():
    relaunched = False
    while True:
        d = datetime.now(TZ)
        if TOKEN:
            tick(d)
            # قبل ما الـjob ده يخلص بـ~12 دقيقة: نطلب commands.yml تاني (بيستنى في الطابور لحد ما ده يخلص)
            if not relaunched and time.time() - T0 > (JOB_MINUTES - 12) * 60:
                try:
                    dispatch("commands.yml", d)
                    relaunched = True
                except Exception as e:  # noqa: BLE001
                    print(f"[ensure] commands relaunch: {str(e)[:120]}", flush=True)
        time.sleep(240)


if __name__ == "__main__":
    main()
