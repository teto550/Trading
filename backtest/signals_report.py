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
if "synthetic" in " ".join(sys.argv) or os.environ.get("SIGNALS_TEST_DROP"):
    # تجربة: منكتبش فوق ملفات الحساب الحقيقية أبدًا
    OUT_DIR = "/tmp/signals_test"
    pa.FILE = os.path.join(OUT_DIR, "account.json")
BOOK_FILE = os.path.join(OUT_DIR, "portfolio.json")
FORECAST = os.path.join(OUT_DIR if OUT_DIR.startswith("/tmp") else os.path.join(ROOT, "forecast"), "swing.json")
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
    live = os.path.join(os.path.dirname(sw.BEST_FILE), "swing_live.json")   # الإعداد الحي المثبّت (بيغلب البحثي)
    src = live if os.path.exists(live) else sw.BEST_FILE
    if os.path.exists(src):
        with open(src, encoding="utf-8") as fh:
            meta = json.load(fh)
        p.update(meta.get("params", {}))
    ov = os.path.join(os.path.dirname(sw.BEST_FILE), "swing_override.json")
    if os.path.exists(ov):                                       # اختيار يدوي موثّق (مثلاً الوقف)
        with open(ov, encoding="utf-8") as fh:
            p.update({k: v for k, v in json.load(fh).items() if not k.startswith("_")})
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
            bday = bars.get(d, {})
            pday = str(D["dates"][tt - 1].date()) if tt > 0 else None
            for msg in pa.adjust_actions(acc, {n: bday[n][5] for n in acc["pos"] if n in bday and bday[n][5]}, d, pday):
                done.append({"side": "note", "name": "", "why": msg, "date": d})
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
                    ps["last_date"] = d
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
                ps["last_date"] = day
    today_hist = [h for h in acc["history"] if h["date"] == day and h not in done]
    # قرارات بكره (نفس swing_core.decide)
    pos_i = {idx[n]: {"rel": ps["last_px"] / ps["entry_px"], "peak": ps["peak_px"] / ps["entry_px"],
                      "atr": ps["atr"], "days": ps["days"]} for n, ps in acc["pos"].items() if n in idx}
    pend = {idx[o["name"]]: o["why"] for o in acc["orders"] if o["side"] == "sell" and o["name"] in idx}
    exits, entries = sc.decide(D, p, t, pos_i, pend)
    eq = pa.equity(acc)
    for i, why in exits.items():
        acc["orders"].append({"side": "sell", "name": names[i], "why": why, "decided": day, "ref_px": float(D["C"][t, i]),
                              "slip": float(D["hs"][t, i]) if p.get("real_cost") else 0.0})
    for i in entries:
        atr_i = float(D["atr"][t, i])
        budget = eq / p["max_pos"]
        if p.get("size", "eq") != "eq":          # حجم حسب المخاطرة (نفس swing_core.exec_open)
            budget = min(eq * float(p["size"]) / ((p["trail_k"] or 2.5) * atr_i), eq * 0.30)
        if p.get("volman"):
            budget *= float(D["vscale"][t])
        acc["orders"].append({"side": "buy", "name": names[i], "why": "signal", "decided": day,
                              "budget": round(budget, 2), "ref_px": float(D["C"][t, i]),
                              "slip": float(D["hs"][t, i]) if p.get("real_cost") else 0.0,
                              "atr": float(D["atr"][t, i]), "score": round(float(D["S"][t, i]), 4)})
    return acc, done + today_hist, exits, entries


PRED_FILE = os.path.join(OUT_DIR, "predictions.json")
TOP_PRED = 5          # عدد الأسهم في "توقعات البوت"
PRED_WINDOW = 20      # بنقيّم توقعات آخر كام جلسة (الموديل بيتوقع لـ20-60 جلسة)


def top_predictions(D, n=TOP_PRED, p=None):
    """أعلى الأسهم في ترتيب البوت اللي مستوفية شرط الاتجاه (لو الإعداد بيطلبه) - دول اللي هيشتري منهم أول ما يقدر."""
    t = len(D["dates"]) - 1
    S = D["S"][t]
    need50 = bool(p and p.get("trend50"))
    ii = [i for i in np.argsort(-np.nan_to_num(S, nan=-1)) if np.isfinite(S[i]) and D["ok_in"][t, i]
          and (not need50 or D["sma50"][t, i])][:n]
    return [str(D["names"][i]).replace(".CA", "") for i in ii]


def update_predictions(D, picks, record):
    """بيسجّل توقعات النهاردة، وبيرجّع نتيجة توقعات آخر PRED_WINDOW جلسة (من أول يوم اتوقع فيه كل سهم لحد النهاردة)."""
    log = {"days": {}}
    if os.path.exists(PRED_FILE):
        with open(PRED_FILE, encoding="utf-8") as fh:
            log = json.load(fh)
    t = len(D["dates"]) - 1
    day = str(D["dates"][t].date())
    names = [str(n).replace(".CA", "") for n in D["names"]]
    idx = {n: i for i, n in enumerate(names)}
    di = {str(d.date()): k for k, d in enumerate(D["dates"])}
    first = {}
    for d in sorted(log["days"]):
        k = di.get(d)
        if k is None or d >= day or t - k > PRED_WINDOW:
            continue
        for n in log["days"][d]:
            first.setdefault(n, (d, k))
    Cf = pd.DataFrame(D["C"]).ffill().values
    mk = np.cumprod(1 + D["mkt"])
    res = []
    for n, (d, k) in first.items():
        i = idx.get(n)
        if i is None or not (np.isfinite(Cf[k, i]) and np.isfinite(Cf[t, i])) or Cf[k, i] <= 0:
            continue
        res.append({"n": n, "from": d, "ret": (Cf[t, i] / Cf[k, i] - 1) * 100, "mkt": (mk[t] / mk[k] - 1) * 100})
    if record:
        log["days"][day] = picks
        log["days"] = {d: log["days"][d] for d in sorted(log["days"])[-120:]}
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(PRED_FILE, "w", encoding="utf-8") as fh:
            json.dump(log, fh, ensure_ascii=False, indent=1)
    return sorted(res, key=lambda r: -r["ret"])


def regime_need(D, reg):
    """قد إيه مؤشر السوق (متوسط الأسهم السايلة) محتاج يطلع في جلسة عشان يعدّي متوسطه وفلتر السوق يفتح."""
    idx = np.asarray(D["idx"], float)
    n = reg if isinstance(reg, int) else 100           # "mix" فيه SMA100؛ "br" بيعتمد على الاتساع
    if reg == "br" or len(idx) < n:
        return None
    x = idx[-(n - 1):].sum() / (n - 1)                 # x > (sum(آخر n-1) + x) / n
    return (x / idx[-1] - 1) * 100


def build_message(D, p, meta, acc, done, now, note, picks=None, pred_res=None):
    """رسالة واحدة مركّزة: توقعات البوت، هيعمل إيه بكره، اللي اتنفذ، التوقعات اللي اتحققت، والمحفظة."""
    t = len(D["dates"]) - 1
    data_date = D["dates"][t].date()
    target = next_session(pd.Timestamp(data_date)).date()
    reg_on = bool(D["reg"][p["regime"]][t])
    fee = acc["fee"]
    names = [str(n).replace(".CA", "") for n in D["names"]]
    idx = {n: i for i, n in enumerate(names)}
    buys = [o for o in acc["orders"] if o["side"] == "buy"]
    sells = [o for o in acc["orders"] if o["side"] == "sell"]
    L_ = [f"🧭 البوت - جلسة {AR_DAYS[target.weekday()]} {target} (على قفلة {data_date})"]
    expect = pd.Timestamp(now.date())
    if not (now.weekday() in TRADING_DAYS and (now.hour, now.minute) >= (15, 0)):
        expect -= pd.Timedelta(days=1)
    while expect.weekday() not in TRADING_DAYS:
        expect -= pd.Timedelta(days=1)
    if pd.Timestamp(data_date) < expect:
        L_.append(f"⚠️ آخر بيانات عندي قفلة {data_date} ومتوقع {expect.date()}: يا إما إجازة رسمية، يا إما مصدر الأسعار "
                  "(TradingView) واقع. لو مش إجازة قولّي.")
    if note:
        L_.append(note)
    L_.append("")
    # 1) التوقعات
    if picks:
        L_.append("🔮 أفضل الأسهم حسب قواعد البوت دلوقتي (مرشحة للشراء في الـ60 جلسة الجاية):")
        bset, hset = {o["name"] for o in buys}, set(acc["pos"])
        for k, n in enumerate(picks, 1):
            i = idx[n]
            tag = " ← 🟢 هيشتريه بكره" if n in bset else (" ← 📂 معانا في المحفظة" if n in hset else "")
            L_.append(f"{k}. {n} | {pa.px(float(D['C'][t, i]))}{tag}")
        if p["regime"] and not reg_on:
            need = regime_need(D, p["regime"])
            if need:
                L_.append(f"📏 عشان البوت يبدأ يشتري: مؤشر السوق محتاج يطلع حوالي {need:+.1f}% من قفلة النهاردة.")
            br = float(D["breadth"][t]) * 100
            L_.append(f"⏸ فلتر السوق ({sw.REG_AR.get(p['regime'], p['regime'])}) قافل، فالبوت مش هيشتري جديد دلوقتي "
                      f"(نسبة الأسهم فوق متوسط 50 يوم: {br:.0f}%).")
        L_.append("")
    # 2) أوامر بكره
    if sells or buys:
        L_.append("📋 هيعمل إيه عند الافتتاح:")
        for o in sells:
            ps = acc["pos"].get(o["name"])
            if ps:
                pnl = ps["shares"] * ps["last_px"] * (1 - fee) - ps["cost"]
                L_.append(f"🔴 بيع {ps['shares']:,} سهم {o['name']} ≈ {pa.money(ps['shares'] * ps['last_px'])} جنيه "
                          f"({'+' if pnl >= 0 else '-'}{pa.money(abs(pnl))}) - {pa.REASON_AR.get(o['why'], o['why'])}")
        for o in buys:
            sh = int(o["budget"] // (o["ref_px"] * (1 + o.get("slip", 0)) * (1 + fee)))
            L_.append(f"🟢 شراء ~{sh:,} سهم {o['name']} × {pa.px(o['ref_px'])} ≈ {pa.money(sh * o['ref_px'])} جنيه")
        L_.append("")
    # 3) اللي اتنفذ
    if done:
        L_.append("🧾 اللي اتنفذ:")
        L_ += [pa.trade_line(r, str(data_date)) for r in done]
        L_.append("")
    # 4) التوقعات اللي اتحققت
    if pred_res:
        up = [r for r in pred_res if r["ret"] > 0]
        avg = float(np.mean([r["ret"] for r in pred_res]))
        mavg = float(np.mean([r["mkt"] for r in pred_res]))
        L_.append(f"✅ نتيجة توقعات آخر {PRED_WINDOW} جلسة: {len(up)} من {len(pred_res)} سهم طلعوا | "
                  f"متوسطهم {avg:+.1f}% والسوق {mavg:+.1f}%")
        for r in up[:8]:
            L_.append(f"   ✅ {r['n']} {r['ret']:+.1f}% (من {r['from']})")
        down = [r for r in pred_res if r["ret"] <= 0]
        if down:
            L_.append("   ❌ نزلوا: " + "، ".join(f"{r['n']} {r['ret']:+.1f}%" for r in down[:8]))
        L_.append("")
    # 5) المحفظة
    if acc["pos"]:
        L_.append(f"📂 المحفظة ({len(acc['pos'])} سهم):")
        for n, ps in sorted(acc["pos"].items(), key=lambda kv: -(kv[1]["last_px"] / kv[1]["entry_px"])):
            pnl = ps["shares"] * ps["last_px"] - ps["cost"]
            L_.append(f"- {n}: {ps['shares']:,} سهم | {pa.px(ps['entry_px'])} ← {pa.px(ps['last_px'])} | "
                      f"{'+' if pnl >= 0 else '-'}{pa.money(abs(pnl))} جنيه ({pnl / ps['cost'] * 100:+.1f}%)")
    eqh = acc["equity"]
    mret = (eqh[-1][2] / eqh[0][2] - 1) * 100 if len(eqh) >= 2 else None
    eq = pa.equity(acc)
    L_.append(f"💼 الحساب: {pa.money(eq)} جنيه ({(eq / acc['capital'] - 1) * 100:+.2f}% من {pa.money(acc['capital'])}) | "
              f"كاش {pa.money(acc['cash'])}" + (f" | السوق {mret:+.1f}%" if mret is not None else ""))
    L_ += ["", "⚠️ حساب تجريبي بفلوس وهمية، مش ضمان مكسب."]
    return "\n".join(L_), target


HIST_H = 40           # إحصائيات "اللي كانوا في نفس الترتيب" بعد كام جلسة


def rank_history(D, H=HIST_H, start="2012-01-01"):
    """لكل عُشر في ترتيب البوت: الأسهم اللي كانت فيه طلعت كام بعد H جلسة (من قفلة اليوم اللي بعده)،
    مقارنة بالسوق، ونسبة المرات اللي طلعت فيها. كله من الماضي الفعلي، مش تقدير."""
    S = D["S"]
    T = len(D["dates"])
    if np.isfinite(S[:-H]).sum() < 5000:
        return None
    lr = np.log1p(D["rc"])
    cs = np.vstack([np.zeros((1, lr.shape[1])), np.cumsum(lr, axis=0)])
    fwd = np.full(S.shape, np.nan)
    fwd[:T - H - 1] = np.exp(cs[H + 2:T + 1] - cs[2:T - H + 1]) - 1        # من قفلة t+1 لقفلة t+1+H
    mk = np.concatenate([[0], np.cumsum(np.log1p(D["mkt"]))])
    mf = np.full(T, np.nan)
    mf[:T - H - 1] = np.exp(mk[H + 2:T + 1] - mk[2:T - H + 1]) - 1
    t0 = int(np.searchsorted(D["dates"], pd.Timestamp(start)))
    out = {}
    for lab, regmask in (("all", np.ones(T, bool)), ("on", D["reg"][100]), ("off", ~D["reg"][100])):
        rows = {}
        for d in range(10):
            m = np.isfinite(S) & np.isfinite(fwd) & D["ok_in"] & (np.floor(np.clip(S, 0, 0.9999) * 10) == d)
            m[:t0] = False
            m &= regmask[:, None]
            v = fwd[m]
            if len(v) < 200:
                continue
            ex = v - np.broadcast_to(mf[:, None], S.shape)[m]
            rows[d] = {"n": int(len(v)), "avg": round(float(v.mean() * 100), 2), "med": round(float(np.median(v) * 100), 2),
                       "win": round(float((v > 0).mean() * 100), 1), "exc": round(float(ex.mean() * 100), 2)}
        out[lab] = rows
    return out


def stock_track(D, p):
    """سجل البوت لكل سهم في الباكتست بنفس القواعد الحية (2012+): عدد الصفقات، كسبانة، متوسط."""
    S = D["S"]
    if np.isfinite(S[:-60]).sum() < 5000:
        return {}
    t0 = int(np.searchsorted(D["dates"], pd.Timestamp("2012-01-01")))
    _e, _x, trades = sc.simulate(D, p, t0, len(D["dates"]))
    out = {}
    for tr in trades:
        if tr.get("open"):
            continue
        n = str(D["names"][tr["i"]]).replace(".CA", "")
        o = out.setdefault(n, {"n": 0, "win": 0, "sum": 0.0, "last": None})
        o["n"] += 1
        o["win"] += int(tr["ret"] > 0)
        o["sum"] += tr["ret"]
        o["last"] = [str(D["dates"][tr["et"]].date()), str(D["dates"][tr["xt"]].date()), round(tr["ret"] * 100, 1)]
    for o in out.values():
        o["avg"] = round(o.pop("sum") / o["n"] * 100, 2)
    return out


def write_forecast(D, p, acc, exits, entries, target):
    t = len(D["dates"]) - 1
    names, S, C = D["names"], D["S"], D["C"]
    try:
        hist = rank_history(D)
        track = stock_track(D, p)
    except Exception as e:  # noqa: BLE001
        print(f"[signals] إحصائيات الترتيب فشلت: {e}")
        hist, track = None, {}
    Cf = pd.DataFrame(C).ffill().values
    status = {}
    for i in range(len(names)):
        n = str(names[i]).replace(".CA", "")
        sc_ = float(S[t, i]) if np.isfinite(S[t, i]) else None
        st = ("دخول" if i in entries else "خروج" if i in exits else "احتفاظ" if n in acc["pos"] else "مراقبة")
        c = float(Cf[t, i]) if np.isfinite(Cf[t, i]) else None
        hi = np.nanmax(Cf[max(0, t - 244):t + 1, i]) if c else None
        status[n] = {"score": round(sc_, 4) if sc_ is not None else None, "status": st, "close": c,
                     "liquid": bool(D["ok_in"][t, i]), "sma50": bool(D["sma50"][t, i]),
                     "atr": round(float(D["atr"][t, i]) * 100, 2), "spread": round(float(D["hs"][t, i]) * 100, 2),
                     "from_hi": round((c / hi - 1) * 100, 1) if c and hi else None,
                     "track": track.get(n)}
    out = {"data_date": str(D["dates"][t].date()), "target": str(target), "params": p,
           "regime_on": bool(D["reg"][p["regime"]][t]), "regime_need": regime_need(D, p["regime"]),
           "breadth": round(float(D["breadth"][t]) * 100, 1), "hist": hist, "hist_h": HIST_H, "stocks": status}
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
        p, meta = load_params()
        kind = (meta or {}).get("scores") or ("sec" if (meta or {}).get("sector") else "ml")
        a.sector = kind == "sec"
        t_ = len(C) - 1
        S = np.full(C.shape, np.nan)
        Sml = today_scores(C, H, L, V, O, liquid, lock, a) if kind in ("ml", "sec", "blend") else None
        if kind in ("ml", "sec"):
            S = Sml
        elif kind == "blend":
            S[t_] = sw.blend_scores(Sml[t_:t_ + 1], sw.alt_scores(C, liquid, "lowvol")[t_:t_ + 1])[0]
        else:
            S = sw.make_scores(kind, C, liquid)          # تاريخ كامل (رخيص) → إحصائيات وسجل البوت لكل سهم
        D = sc.prepare(C, H, L, V, O, S, liquid, lock)
        bars = {}
        if a.source == "yfinance":
            import egxdata
            bars = egxdata.load_bars()
        acc, done, exits, entries = run_account(D, p, bars)
        picks = top_predictions(D, p=p)
        pred_res = update_predictions(D, picks, not a.no_record)
        text, target = build_message(D, p, meta, acc, done, now, note, picks, pred_res)
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
