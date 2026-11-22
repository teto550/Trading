#!/usr/bin/env python3
"""
================================================================
   EGX - إشارات الدخول والخروج اليومية (محرك Swing) + محفظة تجريبية
================================================================
بيشتغل بعد قفل السوق:
  1) بيجيب الأسعار (Yahoo + TradingView للجلسات اللي Yahoo لسه منزّلهاش)
  2) بيدرّب LightGBM على أهداف 20/40/60 يوم على كل التاريخ ويطلع درجة ترتيب لكل سهم النهاردة
  3) بيطبق نفس قواعد swing_core اللي اتختبرت في swing.py (الإعدادات من results/swing_best.json)
     على محفظة تجريبية (signals/portfolio.json): ينفذ قرارات امبارح بافتتاح النهاردة، ويقيّم،
     ويقرر لبكره: ادخل / اخرج / احتفظ - كل قرار بسببه وبمستوى الوقف.
  4) بيبعت رسالة تليجرام واحدة واضحة، وبيكتب forecast/swing.json (للبوت ولتقرير الجلسة).

الاستخدام:
  python backtest/signals_report.py                 # بيبعت تليجرام لو التوكن متحدد
  python backtest/signals_report.py --no-telegram --no-record
  python backtest/signals_report.py --source synthetic-planted --no-telegram   # تجربة من غير نت
================================================================
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import backtest as bt  # noqa: E402
import swing as sw  # noqa: E402
import swing_core as sc  # noqa: E402

OUT_DIR = os.path.join(ROOT, "signals")
BOOK_FILE = os.path.join(OUT_DIR, "portfolio.json")
FORECAST = os.path.join(ROOT, "forecast", "swing.json")
TZ = ZoneInfo("Africa/Cairo")
TRADING_DAYS = {6, 0, 1, 2, 3}
AR_DAYS = {6: "الأحد", 0: "الاتنين", 1: "التلات", 2: "الأربع", 3: "الخميس", 4: "الجمعة", 5: "السبت"}


def next_session(d):
    d = d + pd.Timedelta(days=1)
    while d.weekday() not in TRADING_DAYS:
        d += pd.Timedelta(days=1)
    return d


def fmt_px(x):
    return f"{x:.3f}" if x < 1 else f"{x:.2f}"


def load_params():
    p = dict(sc.DEFAULT)
    meta = None
    if os.path.exists(sw.BEST_FILE):
        with open(sw.BEST_FILE, encoding="utf-8") as fh:
            meta = json.load(fh)
        p.update(meta.get("params", {}))
    if os.environ.get("SWING_PARAMS"):                          # للتجربة: '{"regime": 0}'
        p.update(json.loads(os.environ["SWING_PARAMS"]))
    return p, meta


def today_scores(C, H, L, V, O, liquid, lock, a):
    """درجة كل سهم في آخر يوم: نفس موديل الباكتست (LightGBM × 20/40/60، وزن الأحدث) بس متدرب على كل التاريخ."""
    import improve2 as i2
    import models
    X2, ok2 = sw.features(C, H, L, V, O, liquid, lock, a)
    T, N = C.shape
    t = T - 1
    cols = np.where(ok2[t])[0]
    if len(cols) < 30:
        raise bt.BacktestError(f"عدد الأسهم الصالحة في آخر يوم قليل ({len(cols)})")
    preds = []
    for h in sw.HORIZONS:
        t_ = time.time()
        TGT, E, fv, Y = i2.prep(C, O, h, ok2)
        age = (T - 1 - np.arange(T)).astype(float)
        w = np.repeat((0.5 ** (age / 365.0))[:, None], N, axis=1)[E]
        m = models.make("lgbm", a.trees)
        m.fit(X2[E], Y[E], sample_weight=w)
        p = np.full((T, N), np.nan)
        p[t, cols] = m.predict(X2[t, cols])
        preds.append(p)
        print(f"[signals] موديل {h} يوم اتدرّب على {int(E.sum())} صف في {time.time() - t_:.0f}s", flush=True)
    S = np.full((T, N), np.nan)
    S[t] = sw.rank_mean([p[t:t + 1] for p in preds])[0]
    return S


# ============================================================
# المحفظة التجريبية (بالأسماء، عشان قايمة الأسهم ممكن تتغير بين الأيام)
# ============================================================
def load_book(D):
    if not os.path.exists(BOOK_FILE):
        return None
    with open(BOOK_FILE, encoding="utf-8") as fh:
        raw = json.load(fh)
    name_i = {n: i for i, n in enumerate(D["names"])}
    di = {str(d.date()): i for i, d in enumerate(D["dates"])}
    bk = sc.new_book()
    bk["cash"] = raw["cash"]
    for n, ps in raw["pos"].items():
        if n not in name_i:
            print(f"[signals] {n} مش موجود في البيانات النهاردة، بيتساب في المحفظة من غير تقييم")
            bk.setdefault("orphans", {})[n] = ps
            continue
        q = dict(ps)
        q["et"] = di.get(ps["entry_date"], -1)
        bk["pos"][name_i[n]] = q
    bk["pend_x"] = {name_i[n]: w for n, w in raw.get("pend_x", {}).items() if n in name_i}
    bk["pend_in"] = [name_i[n] for n in raw.get("pend_in", []) if n in name_i]
    bk["carry_x"] = {name_i[n]: w for n, w in raw.get("carry_x", {}).items() if n in name_i}
    bk["meta"] = raw
    return bk


def save_book(bk, D, last_date, closed_now, eq_hist, carry):
    names = D["names"]
    raw = bk.get("meta", {})
    pos = {}
    for i, ps in bk["pos"].items():
        q = {k: v for k, v in ps.items() if k != "et"}
        q["entry_date"] = ps.get("entry_date") or str(D["dates"][ps["et"]].date())
        pos[str(names[i])] = q
    pos.update(bk.get("orphans", {}))
    closed = raw.get("closed", []) + closed_now
    out = {"start": raw.get("start", last_date), "last_date": last_date, "cash": bk["cash"], "pos": pos,
           "pend_x": {str(names[i]): w for i, w in bk["pend_x"].items()},
           "pend_in": [str(names[i]) for i in bk["pend_in"]],
           "carry_x": {str(names[i]): w for i, w in carry.items()},
           "closed": closed[-500:], "equity": eq_hist[-1500:]}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(BOOK_FILE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


def run_book(D, p, record):
    """بيكمّل المحفظة من آخر يوم اتسجل لحد آخر يوم في البيانات، وبيقرر لبكره."""
    T = len(D["dates"])
    t_last = T - 1
    bk = load_book(D)
    names = D["names"]
    closed_now, opened_now = [], []
    if bk is None:
        bk = sc.new_book()
        bk["meta"] = {}
        start_t = t_last                                       # أول تشغيل: نبدأ من النهاردة بقرارات بس
        eq_hist = [[str(D["dates"][t_last].date()), 1.0, 1.0]]
    else:
        last = pd.Timestamp(bk["meta"]["last_date"])
        start_t = int(np.searchsorted(D["dates"], last, side="right"))
        eq_hist = bk["meta"].get("equity", [])
        mk = eq_hist[-1][2] if eq_hist else 1.0
        for t in range(start_t, T):
            out_, in_ = sc.exec_open(D, p, t, bk)
            for tr in out_:
                closed_now.append({"name": str(names[tr["i"]]), "entry_date": str(D["dates"][tr["et"]].date()) if tr["et"] >= 0 else "?",
                                   "exit_date": str(D["dates"][t].date()), "ret": round(tr["ret"] * 100, 2),
                                   "days": tr["days"], "why": tr["why"]})
            for i in in_:
                bk["pos"][i]["entry_date"] = str(D["dates"][t].date())
                opened_now.append(str(names[i]))
            eq, _ex = sc.mark_close(D, t, bk)
            mk *= 1 + D["mkt"][t]
            eq_hist.append([str(D["dates"][t].date()), round(eq, 6), round(mk, 6)])
        if start_t > t_last:
            # تشغيل تاني لنفس القفلة: نرجع لحالة ما قبل قرارات المرة اللي فاتت ونقرر من جديد
            bk["pend_x"] = dict(bk.get("carry_x", {}))
            bk["pend_in"] = []
    carry = dict(bk["pend_x"])                                 # خروج متأجل من قبل كده (السهم كان موقوف)
    exits, entries = sc.decide(D, p, t_last, bk["pos"], bk["pend_x"])
    bk["pend_x"].update(exits)
    bk["pend_in"] = entries
    state = None
    if record:
        state = save_book(bk, D, str(D["dates"][t_last].date()), closed_now, eq_hist, carry)
    else:
        state = {"closed": bk.get("meta", {}).get("closed", []) + closed_now, "equity": eq_hist,
                 "start": bk.get("meta", {}).get("start", str(D["dates"][t_last].date()))}
    bk["carry"] = carry
    return bk, exits, entries, closed_now, opened_now, state


def build_message(D, p, meta, bk, exits, entries, closed_now, opened_now, state, now, note):
    t = len(D["dates"]) - 1
    names, C, S = D["names"], D["C"], D["S"]
    data_date = D["dates"][t].date()
    target = next_session(pd.Timestamp(data_date)).date()
    reg_on = bool(D["reg"][p["regime"]][t])
    L_ = [f"🧭 إشارات الدخول والخروج - لجلسة {AR_DAYS[target.weekday()]} {target}",
          f"(مبنية على قفلة {data_date} | التنفيذ عند افتتاح الجلسة)"]
    if note:
        L_.append(note)
    L_.append("")
    if p["regime"]:
        L_.append(("🟢 السوق في اتجاه صاعد (متوسط الأسهم فوق متوسط " if reg_on else
                   "🔴 السوق في اتجاه هابط (متوسط الأسهم تحت متوسط ") + f"{p['regime']} جلسة)" +
                  (" → مسموح دخول صفقات جديدة" if reg_on else " → مفيش دخول جديد لحد ما يرجع فوقه"))
        L_.append("")

    def px(i):
        return float(C[t, i])

    if entries:
        L_.append(f"🟢 ادخل عند الافتتاح ({len(entries)}):")
        for k, i in enumerate(entries, 1):
            atr = float(D["atr"][t, i])
            stop = px(i) * (1 - p["trail_k"] * atr) if p["trail_k"] else None
            L_.append(f"{k}. {names[i].replace('.CA', '')} | آخر قفلة {fmt_px(px(i))} | ترتيب الموديل: أعلى "
                      f"{max(1, round((1 - S[t, i]) * 100))}% | تذبذب يومي {atr * 100:.1f}%")
            L_.append(f"   متشتريش لو فتح فوق {fmt_px(px(i) * 1.09)} (على الحد الأعلى)"
                      + (f" | وقف مبدئي {fmt_px(stop)} ({-p['trail_k'] * atr * 100:.1f}%)" if stop else ""))
            L_.append(f"   حجم الصفقة: 1/{p['max_pos']} من رأس المال المخصص")
        L_.append("")
    if exits:
        L_.append(f"🔴 اخرج عند الافتتاح ({len(exits)}):")
        for i, why in exits.items():
            ps = bk["pos"][i]
            entry = px(i) / ps["rel"]
            L_.append(f"- {names[i].replace('.CA', '')} | السبب: {sc.REASON_AR.get(why, why)} | "
                      f"دخلنا {ps.get('entry_date', '?')} بسعر ~{fmt_px(entry)} | دلوقتي {fmt_px(px(i))} "
                      f"({(ps['rel'] * (1 - p['cost']) ** 2 - 1) * 100:+.1f}% صافي)")
        L_.append("")
    holds = [i for i in bk["pos"] if i not in bk["pend_x"]]
    if holds:
        L_.append(f"📂 احتفظ ({len(holds)}):")
        for i in sorted(holds, key=lambda j: -bk["pos"][j]["rel"]):
            ps = bk["pos"][i]
            entry = px(i) / ps["rel"]
            stop = entry * ps["peak"] * (1 - p["trail_k"] * ps["atr"]) if p["trail_k"] else None
            L_.append(f"- {names[i].replace('.CA', '')} | من {ps.get('entry_date', '?')} بـ~{fmt_px(entry)} | "
                      f"دلوقتي {fmt_px(px(i))} ({(ps['rel'] - 1) * 100:+.1f}%)"
                      + (f" | الوقف المتحرك {fmt_px(stop)}" if stop else "")
                      + f" | جلسة {ps['days']} من {p['max_hold']}")
        L_.append("")
    pend = [i for i in bk.get("carry", {}) if i not in exits]
    if pend:
        L_.append("⏳ لسه مستني يخرج (السهم كان موقوف): " + ", ".join(names[i].replace(".CA", "") for i in pend))
        L_.append("")
    if not entries and not exits and not holds:
        L_ += ["⚪ مفيش صفقات النهاردة: الموديل مش شايف فرصة بالشروط (ده قرار كمان - الكاش مركز).", ""]
    if opened_now or closed_now:
        L_.append("🧾 اللي اتنفذ في المحفظة التجريبية من آخر تقرير:")
        if opened_now:
            L_.append("• دخول: " + ", ".join(n.replace(".CA", "") for n in opened_now))
        for c in closed_now:
            L_.append(f"• خروج {c['name'].replace('.CA', '')}: {c['ret']:+.1f}% صافي في {c['days']} جلسة ({sc.REASON_AR.get(c['why'], c['why'])})")
        L_.append("")
    eqh = state.get("equity") or []
    closed = state.get("closed") or []
    if len(eqh) >= 2:
        e0, m0 = eqh[0][1], eqh[0][2]
        e1, m1 = eqh[-1][1], eqh[-1][2]
        wins = sum(1 for c in closed if c["ret"] > 0)
        L_.append(f"💼 المحفظة التجريبية من {state.get('start')}: {(e1 / e0 - 1) * 100:+.1f}% | السوق: {(m1 / m0 - 1) * 100:+.1f}%"
                  + (f" | صفقات مقفولة {len(closed)} (كسبانة {wins})" if closed else ""))
    if meta and meta.get("test"):
        m = meta["test"]
        L_.append(f"📊 الباكتست على {m.get('n', 0)} صفقة في 2022+ (فترة الموديل مشافهاش): عائد سنوي {m['cagr']:+.0f}% "
                  f"مقابل {m['mkt_cagr']:+.0f}% للسوق | أقصى هبوط {m['mdd']:.0f}% | كسبانة {m['win']:.0f}% | "
                  f"متوسط المكسب {m['avg_win']:+.1f}% والخسارة {m['avg_loss']:+.1f}%")
    L_ += ["",
           "📌 القواعد (ثابتة ومتختبرة، مش رأي يومي):",
           f"• دخول: السهم من أعلى {round((1 - p['q_in']) * 100)}% في ترتيب الموديل"
           + (" وفوق متوسط 50 يوم" if p["trend50"] else "") + (" والسوق في اتجاه صاعد" if p["regime"] else "") + ".",
           "• خروج: " + ", ".join(x for x in [
               f"وقف متحرك {p['trail_k']}× التذبذب اليومي تحت أعلى سعر من وقت الدخول" if p["trail_k"] else None,
               f"أو ترتيبه نزل تحت {p['q_out']:.0%} (بعد {p['min_hold']} جلسات على الأقل)",
               f"أو عدّى {p['max_hold']} جلسة",
               "أو السوق قلب لهابط" if p["exit_on_regime"] else None] if x) + ".",
           f"• كل صفقة 1/{p['max_pos']} من رأس المال، وأقصى {p['max_pos']} صفقات مع بعض.",
           "⚠️ دي محفظة تجريبية بقواعد إحصائية، مش ضمان مكسب. أي سهم ممكن ينزل، والوقف بيحد الخسارة مش بيمنعها. القرار ليك."]
    return "\n".join(L_), target


def write_forecast(D, p, bk, exits, entries, target):
    t = len(D["dates"]) - 1
    names, S = D["names"], D["S"]
    status = {}
    for i in range(len(names)):
        if not np.isfinite(S[t, i]):
            continue
        st = ("دخول" if i in entries else "خروج" if (i in exits or i in bk["pend_x"]) else
              "احتفاظ" if i in bk["pos"] else "مراقبة")
        status[names[i].replace(".CA", "")] = {"score": round(float(S[t, i]), 4), "status": st}
    out = {"data_date": str(D["dates"][t].date()), "target": str(target), "params": p,
           "regime_on": bool(D["reg"][p["regime"]][t]), "stocks": status}
    os.makedirs(os.path.dirname(FORECAST), exist_ok=True)
    with open(FORECAST, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--min-turnover", type=float, default=1_000_000)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--force-send", action="store_true", help="ابعت الرسالة حتى لو اتبعتت لنفس القفلة")
    a = ap.parse_args()
    now = datetime.now(TZ)
    try:
        if a.fresh:
            cache = os.path.join(bt.DATA_DIR, "prices.pkl")
            if os.path.exists(cache):
                os.remove(cache)
        C, H, L, V, O, liquid, lock = sw.load_data(a)
        drop = int(os.environ.get("SIGNALS_TEST_DROP", "0"))     # للتجربة بس: نشيل آخر كام يوم
        if drop:
            C, H, L, V, O, liquid, lock = (x.iloc[:-drop] for x in (C, H, L, V, O, liquid, lock))
        note = None
        if a.source == "yfinance":
            import egxdata
            if now.weekday() in TRADING_DAYS and (now.hour, now.minute) < (14, 45):
                keep = C.index.date != now.date()
                C, H, L, V, O = (x[keep] for x in (C, H, L, V, O))
            C, H, L, V, O, note = egxdata.extend_panel(C, H, L, V, O, now)
            _f, _m, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, a.min_turnover)
        S = today_scores(C, H, L, V, O, liquid, lock, a)
        D = sc.prepare(C, H, L, V, O, S, liquid, lock)
        p, meta = load_params()
        bk, exits, entries, closed_now, opened_now, state = run_book(D, p, not a.no_record)
        text, target = build_message(D, p, meta, bk, exits, entries, closed_now, opened_now, state, now, note)
        if not a.no_record:
            write_forecast(D, p, bk, exits, entries, target)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "report_latest.md"), "w", encoding="utf-8") as fh:
            fh.write(text)
        err = os.path.join(OUT_DIR, "error_latest.txt")
        if os.path.exists(err):
            os.remove(err)
        print(text)
        sent_file = os.path.join(OUT_DIR, "sent.json")
        sent = {}
        if os.path.exists(sent_file):
            with open(sent_file, encoding="utf-8") as fh:
                sent = json.load(fh)
        dd = str(D["dates"][-1].date())
        if not a.no_telegram and (sent.get("data_date") != dd or a.force_send):
            import tomorrow_report as tr
            tr.send_long(text)
            if not a.no_record:
                with open(sent_file, "w", encoding="utf-8") as fh:
                    json.dump({"data_date": dd, "at": f"{now:%Y-%m-%d %H:%M}"}, fh)
        elif not a.no_telegram:
            print(f"[signals] رسالة قفلة {dd} اتبعتت قبل كده، مش هبعتها تاني")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "error_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M}\n{tb}")
        if not a.no_telegram:
            bt.send_telegram(f"🧭❌ إشارات الدخول والخروج وقعت:\n{type(e).__name__}: {str(e)[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
