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
import paper_account as pa  # noqa: E402

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
    # البيع اللحظي لما السعر يكسر الوقف (طلب ديفيد): بيتقاس في swing.py كمان ("+ بيع لحظي")
    # البيع اللحظي: بس لو الباكتست (على فترة التطوير) قال إنه أحسن من استنى القفلة، وبالمستوى اللي اختاره
    live_extra = (meta or {}).get("live_extra")
    p["intraday_stop"] = live_extra is not None and os.environ.get("PAPER_INTRADAY_STOP", "true").lower() == "true"
    p["live_extra"] = float(live_extra or 0.0)
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
# الحساب التجريبي بالجنيه (paper_account)
# ============================================================
def _raw(D, bars, tt, name, i):
    """أسعار اليوم tt الحقيقية (من أرشيف TradingView لو موجودة، وإلا من البيانات): (افتتاح, أقل, قفلة, قفلة امبارح)."""
    b = bars.get(str(D["dates"][tt].date()), {}).get(name)
    C, O = D["C"], D["O"]
    if b and b[3]:
        return b[0] or b[3], b[2] or b[3], b[3], b[5]
    o = O[tt, i] if np.isfinite(O[tt, i]) else np.nan
    c = C[tt, i]
    prev = C[tt - 1, i] if tt > 0 else np.nan
    lo = c * (1 + D["lc"][tt, i]) / (1 + D["rc"][tt, i]) if np.isfinite(c) else np.nan
    return o, lo, c, prev


def run_account(D, p, bars):
    T = len(D["dates"])
    t = T - 1
    day = str(D["dates"][t].date())
    names = [str(n).replace(".CA", "") for n in D["names"]]
    idx = {n: i for i, n in enumerate(names)}
    acc = pa.load() or pa.new(p["cost"], day)
    done = []
    if acc["last_date"] is None:
        acc["last_date"] = day
        acc["equity"].append([day, acc["cash"], 1.0])
    elif acc["last_date"] < day:
        mk = acc["equity"][-1][2] if acc["equity"] else 1.0
        for tt in range(T):
            d = str(D["dates"][tt].date())
            if d <= acc["last_date"]:
                continue
            raw = {n: _raw(D, bars, tt, n, idx[n]) for n in set(acc["pos"]) | {o["name"] for o in acc["orders"]} if n in idx}
            # 1) أوامر ماتنفذتش لحظيًا: تتنفذ بافتتاح اليوم
            if acc["orders"]:
                opens = {n: (r[0] if np.isfinite(r[0]) else None) for n, r in raw.items()}
                prevs = {n: r[3] for n, r in raw.items()}
                done += pa.fill_orders(acc, opens, prevs, (d, "افتتاح"))
            for n in list(acc["pos"]):
                if n not in raw:
                    continue
                o, lo, c, _pv = raw[n]
                ps = acc["pos"][n]
                # 2) وقف لحظي فاته التشغيل اللحظي: لو أقل سعر لمس الوقف نبيع عنده (نفس الباكتست)
                stop = pa.stop_px(ps, p["trail_k"] + p["live_extra"]) if p["trail_k"] else None
                if p.get("intraday_stop") and stop and ps["entry_date"] < d and np.isfinite(lo) and lo <= stop:
                    price = min(stop, o) if np.isfinite(o) else stop
                    r = pa.sell(acc, n, round(float(price), 3), (d, "أثناء الجلسة"), "stop_live")
                    if r:
                        done.append(r)
                    continue
                # 3) تقييم القفلة
                if np.isfinite(c):
                    ps["last_px"] = float(c)
                    ps["peak_px"] = max(ps["peak_px"], float(c))
                if ps["entry_date"] < d:
                    ps["days"] += 1
            mk *= 1 + D["mkt"][tt]
            acc["equity"].append([d, round(pa.equity(acc), 2), round(mk, 6)])
        acc["last_date"] = day
    else:
        # تشغيل تاني لنفس القفلة: نشيل أوامر المرة اللي فاتت ونقرر من جديد
        acc["orders"] = [o for o in acc["orders"] if o.get("decided") != day]
    # تحديث آخر سعر من بيانات النهاردة (لو التشغيل اللحظي باع/اشترى النهاردة)
    for n, ps in acc["pos"].items():
        if n in idx:
            c = _raw(D, bars, t, n, idx[n])[2]
            if np.isfinite(c):
                ps["last_px"] = float(c)
                ps["peak_px"] = max(ps["peak_px"], float(c))
    today_hist = [h for h in acc["history"] if h["date"] == day and h not in done]
    # قرارات بكره (نفس swing_core.decide)
    pos_i = {idx[n]: {"rel": ps["last_px"] / ps["entry_px"], "peak": ps["peak_px"] / ps["entry_px"],
                      "atr": ps["atr"], "days": ps["days"]} for n, ps in acc["pos"].items() if n in idx}
    pend = {idx[o["name"]]: o["why"] for o in acc["orders"] if o["side"] == "sell" and o["name"] in idx}
    exits, entries = sc.decide(D, p, t, pos_i, pend)
    eq = pa.equity(acc)
    for i, why in exits.items():
        acc["orders"].append({"side": "sell", "name": names[i], "why": why, "decided": day, "ref_px": float(D["C"][t, i])})
    for i in entries:
        acc["orders"].append({"side": "buy", "name": names[i], "why": "signal", "decided": day,
                              "budget": round(eq / p["max_pos"], 2), "ref_px": float(D["C"][t, i]),
                              "atr": float(D["atr"][t, i]), "score": round(float(D["S"][t, i]), 4)})
    return acc, done + today_hist, exits, entries


def build_message(D, p, meta, acc, done, now, note):
    t = len(D["dates"]) - 1
    data_date = D["dates"][t].date()
    target = next_session(pd.Timestamp(data_date)).date()
    reg_on = bool(D["reg"][p["regime"]][t])
    fee = acc["fee"]
    L_ = [f"🧭 الحساب التجريبي - أوامر جلسة {AR_DAYS[target.weekday()]} {target}",
          f"(مبنية على قفلة {data_date} | الأوامر بتتنفذ لوحدها عند الافتتاح، والمحفظة بتتراقب لحظيًا طول الجلسة)"]
    if note:
        L_.append(note)
    L_.append("")
    if p["regime"]:
        L_.append(("🟢 السوق في اتجاه صاعد (متوسط الأسهم فوق متوسط " if reg_on else
                   "🔴 السوق في اتجاه هابط (متوسط الأسهم تحت متوسط ") + f"{p['regime']} جلسة)" +
                  (" → مسموح شراء جديد" if reg_on else " → مفيش شراء جديد لحد ما يرجع فوقه"))
        L_.append("")
    if done:
        L_.append("🧾 اللي اتنفذ من آخر تقرير:")
        L_ += [pa.trade_line(r, str(data_date)) for r in done]
        L_.append("")
    sells = [o for o in acc["orders"] if o["side"] == "sell"]
    buys = [o for o in acc["orders"] if o["side"] == "buy"]
    if sells:
        L_.append(f"🔴 هبيع عند الافتتاح ({len(sells)}):")
        for o in sells:
            ps = acc["pos"].get(o["name"])
            if not ps:
                continue
            amt = ps["shares"] * ps["last_px"]
            pnl = amt * (1 - fee) - ps["cost"]
            L_.append(f"- {o['name']}: {ps['shares']:,} سهم × ~{pa.px(ps['last_px'])} ≈ {pa.money(amt)} جنيه | "
                      f"{'مكسب' if pnl > 0 else 'خسارة'} تقريبًا {pa.money(abs(pnl))} ({pnl / ps['cost'] * 100:+.1f}%) | "
                      f"السبب: {pa.REASON_AR.get(o['why'], o['why'])}")
        L_.append("")
    if buys:
        L_.append(f"🟢 هشتري عند الافتتاح ({len(buys)}) - كل صفقة حوالي {pa.money(buys[0]['budget'])} جنيه:")
        for o in buys:
            sh = int(o["budget"] // (o["ref_px"] * (1 + fee)))
            stop = o["ref_px"] * (1 - p["trail_k"] * o["atr"]) if p["trail_k"] else None
            L_.append(f"- {o['name']}: ~{sh:,} سهم × ~{pa.px(o['ref_px'])} ≈ {pa.money(sh * o['ref_px'])} جنيه | "
                      f"ترتيب الموديل أعلى {max(1, round((1 - o['score']) * 100))}%"
                      + (f" | وقف مبدئي {pa.px(stop)} ({-p['trail_k'] * o['atr'] * 100:.1f}%)" if stop else "")
                      + " | لو فتح على الحد الأعلى (+9%) مش هشتري")
        L_.append("")
    if acc["pos"]:
        L_.append(f"📂 المحفظة دلوقتي ({len(acc['pos'])} سهم):")
        for n, ps in sorted(acc["pos"].items(), key=lambda kv: -(kv[1]["last_px"] / kv[1]["entry_px"])):
            val = ps["shares"] * ps["last_px"]
            pnl = val - ps["cost"]
            stop = pa.stop_px(ps, p["trail_k"])
            L_.append(f"- {n}: {ps['shares']:,} سهم | شراء {pa.px(ps['entry_px'])} يوم {ps['entry_date']} | دلوقتي {pa.px(ps['last_px'])} "
                      f"= {pa.money(val)} جنيه ({'+' if pnl >= 0 else '-'}{pa.money(abs(pnl))}، {pnl / ps['cost'] * 100:+.1f}%)"
                      + (f" | الوقف {pa.px(stop)}" if stop else "") + f" | جلسة {ps['days']} من {p['max_hold']}")
        L_.append("")
    if not buys and not sells and not acc["pos"]:
        L_ += ["⚪ مفيش أوامر لبكره والحساب كله كاش: الموديل مش شايف فرصة بالشروط (ده قرار كمان).", ""]
    eqh = acc["equity"]
    mret = (eqh[-1][2] / eqh[0][2] - 1) * 100 if len(eqh) >= 2 else None
    L_ += pa.summary_lines(acc, None, mret)
    if meta and meta.get("test"):
        m = meta["test"]
        L_.append(f"📊 الباكتست على {m.get('n', 0)} صفقة في 2022+ (فترة الموديل مشافهاش): عائد سنوي {m['cagr']:+.0f}% "
                  f"مقابل {m['mkt_cagr']:+.0f}% للسوق | أقصى هبوط {m['mdd']:.0f}% | كسبانة {m['win']:.0f}%")
    L_ += ["",
           "📌 القواعد (ثابتة ومتختبرة):",
           f"• شراء: السهم من أعلى {round((1 - p['q_in']) * 100)}% في ترتيب الموديل"
           + (" وفوق متوسط 50 يوم" if p["trend50"] else "") + (" والسوق في اتجاه صاعد" if p["regime"] else "")
           + f"، بـ1/{p['max_pos']} من قيمة الحساب، وأقصى {p['max_pos']} أسهم مع بعض.",
           "• بيع: " + "، ".join(x for x in [
               (f"لو قفل تحت الوقف المتحرك ({p['trail_k']}× التذبذب اليومي تحت أعلى قفلة من وقت الشراء) بيتباع الافتتاح اللي بعده"
                + (f"، ولو انهار تحته بـ{p['live_extra']:.0f}× التذبذب أثناء الجلسة بيتباع فورًا" if p.get("intraday_stop") else
                   "، وأثناء الجلسة بيجيلك تنبيه لحظي أول ما يكسره")) if p["trail_k"] else None,
               f"أو ترتيبه نزل تحت {p['q_out']:.0%}", f"أو عدّى {p['max_hold']} جلسة",
               "أو السوق قلب لهابط" if p["exit_on_regime"] else None] if x) + ".",
           f"• عمولة {fee * 100:.1f}% في كل عملية محسوبة في الأرقام.",
           "⚠️ حساب تجريبي بفلوس وهمية. مش ضمان مكسب، والوقف بيحد الخسارة مش بيمنعها."]
    return "\n".join(L_), target


def write_forecast(D, p, acc, exits, entries, target):
    t = len(D["dates"]) - 1
    names, S = D["names"], D["S"]
    status = {}
    for i in range(len(names)):
        if not np.isfinite(S[t, i]):
            continue
        n = str(names[i]).replace(".CA", "")
        st = ("دخول" if i in entries else "خروج" if i in exits else "احتفاظ" if n in acc["pos"] else "مراقبة")
        status[n] = {"score": round(float(S[t, i]), 4), "status": st}
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
        bars = {}
        if a.source == "yfinance":
            import egxdata
            bars = egxdata.load_bars()
        acc, done, exits, entries = run_account(D, p, bars)
        text, target = build_message(D, p, meta, acc, done, now, note)
        if not a.no_record:
            pa.save(acc)
            write_forecast(D, p, acc, exits, entries, target)
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
