#!/usr/bin/env python3
"""ملخص أسبوعي واحد بدل متابعة كل حاجة يدويًا: النتايج الفعلية لكل أجزاء البوت وهل فيه دليل إنه بيكسب."""
import json
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
STATE = os.path.join(ROOT, "horizon", "weekly_state.json")


def load(p, default=None):
    try:
        with open(os.path.join(ROOT, p), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def avg_vs(days):
    if not days:
        return None
    a = np.mean([d["avg"] for d in days])
    m = np.mean([d["mkt_avg"] for d in days])
    beat = sum(d["avg"] > d["mkt_avg"] for d in days)
    return a, m, beat, len(days)


def models_avg(days):
    acc = {}
    for d in days:
        for k, v in (d.get("models") or {}).items():
            acc.setdefault(k, []).append(v["avg"] if isinstance(v, dict) else v)
    return {k: float(np.mean(v)) for k, v in acc.items() if len(v) >= 1}, max((len(v) for v in acc.values()), default=0)


def build(now):
    L = [f"📊 ملخص أسبوعي - {now:%Y-%m-%d}", "النتايج الفعلية للتوقعات اللي اتسجلت قبل ما النتيجة تظهر (قبل تكلفة ~0.4% للصفقة).", ""]
    good = []

    pp = load("paper/picks.json", {})
    st = pp.get("last_stats") or {}
    if st:
        L.append(f"1) Paper Trade (احتفاظ 20 يوم) - {st.get('n', 0)} دفعة ناضجة من {len(pp.get('cohorts', []))}:")
        L.append(f"   السوق: {st.get('mkt_pct', 0):+.2f}% للدفعة")
        for key, name in (("xgb", "XGBoost"), ("lgbm", "LightGBM+Alpha158"), ("lowvol", "Low-vol")):
            s = st.get(key)
            if s:
                L.append(f"   {name}: {s['ex_pct']:+.2f}% عن السوق، {s['net_pct']:+.2f}% بعد التكلفة، كسب في {s['hit'] * 100:.0f}% من الدفعات")
                if st.get("n", 0) >= 10 and s["net_pct"] > 0 and s["hit"] >= 0.55:
                    good.append(name)
        L.append("")

    t = (load("tomorrow/scorecard.json", {}) or {}).get("days", [])
    r_all, r_wk = avg_vs(t), avg_vs(t[-5:])
    if r_all:
        L.append("2) اختيارات الجلسة الجاية:")
        if r_wk:
            L.append(f"   آخر {r_wk[3]} جلسات: {r_wk[0]:+.2f}% مقابل السوق {r_wk[1]:+.2f}% (غلبت في {r_wk[2]})")
        L.append(f"   من الأول ({r_all[3]} جلسة): {r_all[0]:+.2f}% مقابل {r_all[1]:+.2f}% (غلبت في {r_all[2]})")
        ma, n = models_avg([d for d in t if d.get("models")])
        if ma:
            L.append(f"   أحسن موديل ({n} جلسة فيها بيانات): " + " | ".join(f"{k} {v:+.2f}%" for k, v in sorted(ma.items(), key=lambda x: -x[1])))
        L.append("")

    h = (load("horizon/scorecard.json", {}) or {}).get("days", [])
    if h:
        L.append("3) الفترات الأطول:")
        for hh in sorted({d["h"] for d in h}):
            r = avg_vs([d for d in h if d["h"] == hh])
            if r:
                L.append(f"   {hh} يوم ({r[3]} نتيجة): {r[0]:+.2f}% مقابل السوق {r[1]:+.2f}% (غلبت في {r[2]})")
                if hh >= 20 and r[3] >= 8 and r[2] / r[3] >= 0.6 and r[0] > r[1]:
                    good.append(f"{hh} يوم")
        ma, n = models_avg([d for d in h if d.get("models")])
        if ma:
            L.append(f"   أحسن موديل ({n} نتيجة): " + " | ".join(f"{k} {v:+.2f}%" for k, v in sorted(ma.items(), key=lambda x: -x[1])))
        L.append("")

    lv = load("live_scorecard.json", [])
    if lv:
        n = sum(x["n"] for x in lv)
        hit = sum(x["hit"] * x["n"] for x in lv) / n
        ex = sum(x["excess"] * x["n"] for x in lv) / n
        L += [f"4) التوقع اللحظي ({len(lv)} يوم، {n} توقع، مقفول من الرسايل): اتجاه صح {hit:.0f}% | زيادة عن السوق {ex:+.2f}%", ""]

    ol = [x for x in (load("horizon/outlook_log.json", []) or []) if x.get("done")]
    if ol:
        L.append(f"5) مرجع الدخول: {len(ol)} يوم اتقيّم جزئيًا (التفاصيل في رسايل '🏁 نتيجة مرجع الدخول').")
        L.append("")

    if good:
        L.append("✅ مؤشرات مبشرة (لسه عدد صغير): " + "، ".join(good))
    else:
        L.append("⚪ الخلاصة: لسه مفيش دليل فعلي إن البوت بيكسب بعد التكلفة.")
    L += ["", "معالم جاية: دفعات الـPaper Trade تنضج لحد حوالي 25 نوفمبر، وأول نتيجة حقيقية لـ20 يوم حوالي 9 ديسمبر.",
          "ده ملخص أرقام، مش توصية شراء أو بيع."]
    return "\n".join(L)


def main():
    now = datetime.now(ZoneInfo("Africa/Cairo"))
    force = "--force" in sys.argv
    last = (load("horizon/weekly_state.json", {}) or {}).get("last")
    if not force and last and (now.date() - datetime.strptime(last, "%Y-%m-%d").date()) < timedelta(days=6):
        print(f"[weekly] اتبعت {last} - تخطي")
        return
    msg = build(now)
    print(msg)
    if "--no-telegram" not in sys.argv:
        import backtest as bt
        bt.send_telegram(msg)
    if "--no-record" not in sys.argv:
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump({"last": f"{now:%Y-%m-%d}"}, f)


if __name__ == "__main__":
    main()
