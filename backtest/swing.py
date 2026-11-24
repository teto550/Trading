#!/usr/bin/env python3
"""
================================================================
   EGX Swing Backtest - محفظة حقيقية بقواعد دخول وخروج صريحة
================================================================
بدل "ترتيب أسهم كل يوم"، ده بيختبر طريقة تداول كاملة زي ما هتنفذها فعلاً:
  1) درجات الموديل walk-forward (LightGBM على 20/40/60 يوم + Alpha158، بيتدرّب على الماضي بس،
     ويتعاد تدريبه كل 63 يوم) → متوسط ترتيب الـ3.
  2) قواعد swing_core: دخول بافتتاح اليوم التالي، وقف متحرك بالتذبذب، خروج بالترتيب/المدة/السوق.
  3) اختيار الإعدادات على فترة التطوير (2012-2021) بس، والحكم على 2022+ اللي الإعدادات مشافتهاش.
  4) مقارنة بنفس القواعد على درجات بديلة (عشوائي / تذبذب منخفض) عشان نعرف الموديل نفسه بيضيف ولا لأ.

الاستخدام:
  python backtest/swing.py                      # بيبني الدرجات (أو بيقرا الكاش) وبيعمل الاختبار
  python backtest/swing.py --rebuild            # يعيد تدريب الدرجات
  python backtest/swing.py --source synthetic-planted   # اختبار سلامة سريع
================================================================
"""
import argparse
import itertools
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as bt  # noqa: E402
import swing_core as sc  # noqa: E402

DEV = (pd.Timestamp("2012-01-01"), pd.Timestamp("2022-01-01"))
TEST = (pd.Timestamp("2022-01-01"), pd.Timestamp("2100-01-01"))
PRED_FILE = os.path.join(bt.RES_DIR, "swing_preds.npz")
BEST_FILE = os.path.join(bt.RES_DIR, "swing_best.json")
HORIZONS = (20, 40, 60)

GRID = dict(q_in=[0.90, 0.95], q_out=[0.30, 0.50], max_pos=[5, 8], trail_k=[0.0, 2.5, 3.5],
            max_hold=[40, 60], regime=[0, 50, 100, 200, "br", "mix"], trend50=[False, True],
            exit_on_regime=[False, True], size=["eq", 0.01, 0.02])
REG_AR = {0: "مفيش", 50: "SMA50", 100: "SMA100", 200: "SMA200", "br": "اتساع السوق (>50% فوق SMA50)",
          "mix": "SMA100 أو اتساع قوي (>60%)"}
SIZE_AR = {"eq": "بالتساوي", 0.01: "مخاطرة 1% من الحساب لكل صفقة", 0.02: "مخاطرة 2% من الحساب لكل صفقة"}
SECTORS_FILE = os.path.join(os.path.dirname(HERE), "data", "sectors.json")


def pred_file(sector):
    return os.path.join(bt.RES_DIR, "swing_preds_sec.npz" if sector else "swing_preds.npz")


def robust_pick(results, keys):
    """اختيار ضد الـoverfitting (على التطوير بس): درجة كل إعداد = متوسط Sharpe بتاعه وبتاع كل الإعدادات اللي
    بتختلف عنه في قيمة واحدة بس. كده بنختار منطقة كويسة مش نقطة محظوظة."""
    idx = {tuple(r[0][k] for k in keys): r for r in results}
    best, best_s = None, -1e9
    for kv, r in idx.items():
        if not r[1] or r[1]["n"] < 80:
            continue
        nb = [r[1]["sharpe"]]
        for j, k in enumerate(keys):
            for v in GRID[k]:
                if v == kv[j]:
                    continue
                q = idx.get(kv[:j] + (v,) + kv[j + 1:])
                if q and q[1]:
                    nb.append(q[1]["sharpe"])
        sc_ = float(np.mean(nb))
        if sc_ > best_s:
            best, best_s = r, sc_
    return best, best_s


def load_data(a):
    if a.source == "yfinance":
        panel = bt.fetch_yfinance(bt.load_tickers())
    else:
        import backtest_horizons as bh
        panel = bh.make_synthetic_h(a.source == "synthetic-planted")
    C, H, L, V, O = bt.clean_panel(panel)
    _f, _m, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, a.min_turnover)
    return C, H, L, V, O, liquid, lock


def features(C, H, L, V, O, liquid, lock, a):
    """مصفوفة المدخلات (T,N,K) ومين صالح: المدخلات الأساسية + EGX30/الدولار + Alpha158."""
    import alpha as al
    import improve as im
    import macro
    _f, mkt0, *_ = bt.build_features(C, H, L, V, O, a.min_turnover)
    if a.source == "yfinance":
        M = macro.fetch_macro(fresh=True)
    else:
        M = macro.synthetic_macro(mkt0["mkt1"])
    M, _src = macro.ensure_egx30(M, C, V)
    X, _ok = im.setup(C, H, L, V, O, a.min_turnover, M)
    XA, _names = al.build_alpha(C, H, L, V, O, liquid, bt.csrank)
    parts = [X, XA]
    if getattr(a, "sector", False):
        parts.append(sector_features(C, liquid))
    X2 = np.concatenate(parts, axis=2)
    ok2 = np.isfinite(X2).all(axis=2) & liquid.values & ~lock.values
    return X2, ok2


def sector_features(C, liquid):
    """أداء السهم مقارنة بقطاعه + زخم القطاع نفسه (20 و60 جلسة)، ترتيب نسبي بين الأسهم. القطاع ثابت فمفيش غش."""
    try:
        with open(SECTORS_FILE, encoding="utf-8") as fh:
            secmap = json.load(fh)
    except (OSError, ValueError):
        secmap = {}
    g = np.array([secmap.get(c.replace(".CA", ""), {}).get("sector", "?") for c in C.columns])
    r = (C / C.shift(1) - 1).where(lambda x: x.abs() <= 0.25)
    lr = np.log1p(r.fillna(0))
    out = []
    for h in (20, 60):
        R = (lr.rolling(h, min_periods=h // 2).sum()).where(liquid).values
        M = np.full_like(R, np.nan)
        for sname in set(g):
            m = g == sname
            if sname == "?" or m.sum() < 3:
                continue
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    M[:, m] = np.nanmean(R[:, m], axis=1, keepdims=True)
        for F in (R - M, M):
            Fr = bt.csrank(pd.DataFrame(F, index=C.index, columns=C.columns), liquid).values
            out.append(np.where(np.isfinite(Fr), Fr, 0.0)[:, :, None])
    print(f"[swing] مدخلات القطاعات: {len(set(g) - {'?'})} قطاع، {int((g == '?').sum())} سهم من غير قطاع")
    return np.concatenate(out, axis=2).astype(np.float32)


def rank_mean(preds):
    st = np.stack([pd.DataFrame(p).rank(axis=1, pct=True).values for p in preds])
    S = np.where(np.isfinite(st).all(axis=0), st.mean(axis=0), np.nan)
    return pd.DataFrame(S).rank(axis=1, pct=True).values     # نرجّعها توزيع 0..1 تاني


def build_scores(C, H, L, V, O, liquid, lock, a):
    """درجات walk-forward: ترتيب متوسط LightGBM 20/40/60 (نفس اللي في final_test)."""
    import improve2 as i2
    X2, ok2 = features(C, H, L, V, O, liquid, lock, a)
    preds = []
    for h in HORIZONS:
        t = time.time()
        TGT, E, fv, Y = i2.prep(C, O, h, ok2)
        p, _t0 = i2.walk("lgbm", X2, Y, E, fv, a, h - 1)
        print(f"[swing] درجات h={h} خلصت في {time.time() - t:.0f}s", flush=True)
        preds.append(np.where(ok2, p, np.nan))
    return rank_mean(preds)


VAR_AR = {"ml": "الموديل (أساسي)", "sec": "الموديل + القطاعات", "lowvol": "تذبذب منخفض (من غير موديل)",
          "blend": "الموديل + تذبذب منخفض (متوسط الترتيب)"}


def blend_scores(Sml, Slv):
    """متوسط ترتيب الموديل وترتيب التذبذب المنخفض (لو واحد ناقص ناخد التاني)."""
    st = np.stack([Sml, Slv])
    S = np.where(np.isfinite(Sml) & np.isfinite(Slv), np.nanmean(st, axis=0), np.nan)
    return pd.DataFrame(S).rank(axis=1, pct=True).values


def alt_scores(C, liquid, kind, seed=0):
    if kind == "random":
        rng = np.random.default_rng(seed)
        R = pd.DataFrame(rng.random(C.shape), index=C.index, columns=C.columns)
        R = R.rolling(20, min_periods=1).mean()            # عشوائي بس ثابت نسبيًا (عشان ميعملش churn)
    else:  # lowvol
        r = (C / C.shift(1) - 1).where(lambda x: x.abs() <= 0.25)
        R = -r.rolling(60, min_periods=40).std()
    return R.where(liquid).rank(axis=1, pct=True).values


def fmt(m):
    if not m:
        return "| - | - | - | - | - | - | - | - | - |"
    return (f"| {m['cagr']:+.1f}% | {m['mkt_cagr']:+.1f}% | {m['mdd']:.1f}% ({m['mkt_mdd']:.0f}%) | {m['sharpe']:.2f} | "
            f"{m['n']} | {m['win']:.0f}% | {m['avg_win']:+.1f}% / {m['avg_loss']:+.1f}% | {m['pf']:.2f} | "
            f"{m['hold']:.0f} | {m['expo']:.0f}% |")


HEAD = ["| عائد سنوي | السوق سنوي | أقصى هبوط (السوق) | Sharpe | صفقات | كسبانة | متوسط مكسب / خسارة | Profit factor | أيام احتفاظ | وقت في السوق |",
        "|---|---|---|---|---|---|---|---|---|---|"]


def pdesc(p):
    return (f"أعلى {round((1 - p['q_in']) * 100)}% دخول، خروج تحت {p['q_out']:.0%}، {p['max_pos']} مراكز، "
            f"وقف {p['trail_k']}×ATR{'' if p['trail_k'] else ' (من غير)'}، أقصى {p['max_hold']} يوم، "
            f"فلتر السوق {REG_AR.get(p['regime'], p['regime'])}، فوق SMA50: {'آه' if p['trend50'] else 'لأ'}، "
            f"خروج لما السوق يقلب: {'آه' if p['exit_on_regime'] else 'لأ'}، الحجم: {SIZE_AR.get(p.get('size', 'eq'))}")


def get_scores(C, H, L, V, O, liquid, lock, a, sector):
    f = pred_file(sector)
    if not a.rebuild and a.source == "yfinance" and os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        cols = list(z["cols"])
        miss = [c for c in C.columns if c not in cols]
        if len(miss) <= 2:
            d0 = pd.DatetimeIndex(z["dates"])
            print(f"[swing] درجات {'القطاعات' if sector else 'الأساسية'} من الكاش لحد {d0[-1].date()}")
            return pd.DataFrame(z["S"].astype(float), index=d0, columns=cols).reindex(index=C.index, columns=C.columns).values
        print(f"[swing] الكاش ناقصه {len(miss)} سهم جديد → إعادة تدريب")
    a.sector = sector
    S = build_scores(C, H, L, V, O, liquid, lock, a)
    if a.source == "yfinance":
        os.makedirs(bt.RES_DIR, exist_ok=True)
        np.savez_compressed(f, S=S.astype(np.float16), dates=np.array(C.index.astype(str)), cols=np.array(C.columns))
    return S


def run_grid(D, first, T, dev, test, base_cost):
    keys = list(GRID)
    results = []
    for vals in itertools.product(*[GRID[k] for k in keys]):
        p = dict(sc.DEFAULT, **dict(zip(keys, vals)), cost=base_cost, real_cost=True)
        eq, ex, tr = sc.simulate(D, p, first, T)
        results.append((p, sc.metrics(D, eq, ex, tr, *dev), sc.metrics(D, eq, ex, tr, *test)))
    return results, keys


def run(a):
    C, H, L, V, O, liquid, lock = load_data(a)
    T, N = C.shape
    base_cost = a.base_bps / 1e4
    Sml = get_scores(C, H, L, V, O, liquid, lock, a, False)
    Slv = alt_scores(C, liquid, "lowvol")
    Ss = {"ml": Sml, "lowvol": Slv, "blend": blend_scores(Sml, Slv)}
    if a.source == "yfinance" or os.path.exists(SECTORS_FILE):
        Ss["sec"] = get_scores(C, H, L, V, O, liquid, lock, a, True)
    S0 = Sml
    first = int(np.where(np.isfinite(S0).sum(axis=1) >= 30)[0][0])
    Ds = {k: sc.prepare(C, H, L, V, O, S, liquid, lock) for k, S in Ss.items()}
    D0 = Ds["ml"]
    dev = DEV if a.source == "yfinance" else (C.index[first], C.index[first + (T - first) * 2 // 3])
    test = TEST if a.source == "yfinance" else (dev[1], pd.Timestamp("2100-01-01"))
    print(f"[swing] أول يوم بدرجات: {C.index[first].date()} | تطوير {dev[0].date()}→{dev[1].date()} | اختبار {test[0].date()}+")

    def mt(D, p):
        e, x, t = sc.simulate(D, p, first, T)
        return sc.metrics(D, e, x, t, *dev), sc.metrics(D, e, x, t, *test), (e, x, t)

    # (1) الإعداد القديم: بتكلفة ثابتة 20 bps، وبالتكلفة الحقيقية (عمولة + فرق سعر الشراء/البيع لكل سهم)
    # الإعداد اللي كان شغال لايف قبل التحديث ده (23/11/2026)
    old_p = dict(sc.DEFAULT, q_in=0.90, q_out=0.30, max_pos=5, trail_k=2.5, max_hold=40, regime=100,
                 trend50=False, exit_on_regime=True, size="eq")
    old_flat = mt(D0, dict(old_p, cost=0.002, real_cost=False))
    old_real = mt(D0, dict(old_p, cost=base_cost, real_cost=True))
    hs_test = D0["hs"][(D0["dates"] >= test[0])]
    # (4)+(5)+(6): الشبكة بالتكلفة الحقيقية، لكل نوع مدخلات
    picks = {}
    t0 = time.time()
    for k, D in Ds.items():
        res, keys = run_grid(D, first, T, dev, test, base_cost)
        (bp, bd, bt_), nb = robust_pick(res, keys)
        picks[k] = (bp, bd, bt_, nb, res)
        print(f"[swing] {VAR_AR[k]}: {len(res)} إعداد، أفضل منطقة Sharpe {nb:.2f}", flush=True)
    print(f"[swing] الشبكة خلصت في {time.time() - t0:.0f}s")
    use_sec = max(picks, key=lambda k: picks[k][3])
    best_p, best_dev, best_test, nb_sharpe, results = picks[use_sec]
    D = Ds[use_sec]
    ok = sorted([r for r in results if r[1] and r[1]["n"] >= 80], key=lambda r: -r[1]["sharpe"])
    eq, ex, tr = sc.simulate(D, best_p, first, T)
    yr = sc.yearly(D, eq, C.index[first])
    # فلاتر السوق وطرق الحجم على الإعداد المختار
    reg_rows = [(r, *mt(D, dict(best_p, regime=r))[:2]) for r in GRID["regime"]]
    size_rows = [(z, *mt(D, dict(best_p, size=z))[:2]) for z in GRID["size"]]
    stop_rows = [(k_, *mt(D, dict(best_p, trail_k=k_))[:2]) for k_ in (0.0, 2.5, 3.5, 5.0)]
    hi_cost = mt(D, dict(best_p, cost=base_cost + 0.0015))
    delay = mt(D, dict(best_p, exec_close=True))
    # بيع لحظي
    live_rows, live_best, live_dev_best = [], None, best_dev["sharpe"]
    for ex_ in (0.0, 1.0, 2.0):
        md_, mt_, _ = mt(D, dict(best_p, intraday_stop=True, live_extra=ex_))
        live_rows.append((ex_, md_, mt_))
        if md_ and md_["sharpe"] > live_dev_best:
            live_best, live_dev_best = ex_, md_["sharpe"]
    alts = {}
    for kind in ("random", "lowvol"):
        D2 = sc.prepare(C, H, L, V, O, alt_scores(C, liquid, kind), liquid, lock)
        alts[kind] = mt(D2, best_p)[:2]
    reasons = pd.Series([x["why"] for x in tr if test[0] <= C.index[x["et"]]]).value_counts()

    R = lambda lab, m: f"| {lab} " + fmt(m)  # noqa: E731
    H2 = ["| الإعداد " + HEAD[0], "|---" + HEAD[1]]
    out = [f"# EGX Swing Backtest - {datetime.now():%Y-%m-%d %H:%M}", "",
           f"- الأسهم {N} | {C.index[0].date()} → {C.index[-1].date()} | شراء وبيع بافتتاح الجلسة اللي بعد الإشارة | "
           "السوق = متوسط الأسهم السايلة (equal-weight)",
           f"- **التكلفة الحقيقية**: عمولة {a.base_bps:.0f} bps + نص فرق سعر الشراء/البيع المقدّر لكل سهم كل يوم "
           f"(Abdi-Ranaldo من أعلى/أقل/قفلة، وأقل حاجة نص تيك). متوسطها في 2022+: {np.mean(hs_test) * 1e4:.0f} bps، "
           f"والوسيط {np.median(hs_test) * 1e4:.0f} bps، وأعلى 10% من الأسهم: {np.percentile(hs_test, 90) * 1e4:.0f} bps في كل اتجاه.",
           f"- الإعدادات اتختارت على {dev[0].year}-{dev[1].year - 1} بس (الإعداد اللي هو وجيرانه أعلى متوسط Sharpe = {nb_sharpe:.2f}). "
           f"**{test[0].year}+ محدش شافها.**", "",
           "## (1) تأثير التكلفة الحقيقية على الإعداد القديم", "", pdesc(old_p), "", *H2,
           R("تطوير - تكلفة ثابتة 20 bps (القديم)", old_flat[0]), R("اختبار - تكلفة ثابتة 20 bps (القديم)", old_flat[1]),
           R("تطوير - تكلفة حقيقية", old_real[0]), R("**اختبار - تكلفة حقيقية**", old_real[1]), "",
           "## الإعداد الجديد المختار (بالتكلفة الحقيقية)", "",
           f"الدرجات: {VAR_AR[use_sec]} (اتختارت على التطوير بس من: {'، '.join(VAR_AR[k] for k in picks)})", "",
           pdesc(best_p), "", *H2,
           R(f"تطوير {dev[0].year}-{dev[1].year - 1}", best_dev), R(f"**اختبار {test[0].year}+**", best_test),
           R("اختبار + عمولة أعلى بـ15 bps", hi_cost[1]), R("اختبار + تنفيذ بقفلة اليوم التالي", delay[1]), "",
           "## (6) أنواع الدرجات: أفضل إعداد لكل نوع (الاختيار بـSharpe المنطقة على التطوير)", "", *H2]
    for k in picks:
        out += [R(f"{VAR_AR[k]} - تطوير (Sharpe المنطقة {picks[k][3]:.2f})", picks[k][1]),
                R(f"{VAR_AR[k]} - اختبار", picks[k][2])]
    out += ["", "## (4) فلتر السوق (نفس الإعداد، الفلتر بس بيتغير)", "", *H2]
    for r, md_, mt_ in reg_rows:
        out += [R(f"{REG_AR[r]} - تطوير", md_), R(f"{REG_AR[r]} - اختبار", mt_)]
    out += ["", "## (5) حجم الصفقة (نفس الإعداد، الحجم بس بيتغير)", "", *H2]
    for z, md_, mt_ in size_rows:
        out += [R(f"{SIZE_AR[z]} - تطوير", md_), R(f"{SIZE_AR[z]} - اختبار", mt_)]
    out += ["", "## الوقف المتحرك (نفس الإعداد، الوقف بس بيتغير)", "", *H2]
    for k_, md_, mt_ in stop_rows:
        lab = "من غير وقف" if not k_ else f"وقف {k_}×ATR"
        out += [R(f"{lab} - تطوير", md_), R(f"{lab} - اختبار", mt_)]
    out += ["", "## بيع لحظي أثناء الجلسة", "", *H2]
    for ex_, md_, mt_ in live_rows:
        lab = "عند الوقف" if not ex_ else f"تحت الوقف بـ{ex_:.0f}×ATR"
        out += [R(f"تطوير - {lab}", md_), R(f"اختبار - {lab}", mt_)]
    out += ["", "الاختيار: " + ("مفيش بيع لحظي أحسن من استنى القفلة (تنبيه لحظي بس)." if live_best is None else
                                f"بيع لحظي تحت الوقف بـ{live_best:.0f}×ATR."), "",
            "## نفس القواعد بدرجات تانية", "", *H2, R("المختار - تطوير", best_dev), R("المختار - اختبار", best_test),
            R("عشوائي - تطوير", alts["random"][0]), R("عشوائي - اختبار", alts["random"][1]),
            R("الموديل (ML) - تطوير", mt(Ds["ml"], best_p)[0]), R("الموديل (ML) - اختبار", mt(Ds["ml"], best_p)[1]),
            "", "## سنة بسنة (الإعداد المختار)", "", "| السنة | المحفظة | السوق | الفرق |", "|---|---|---|---|"]
    out += [f"| {y} | {s_:+.1f}% | {m:+.1f}% | {s_ - m:+.1f}% |" for y, s_, m in yr]
    out += ["", "## أسباب الخروج في الاختبار", ""] + [f"- {sc.REASON_AR.get(k, k)}: {v}" for k, v in reasons.items()]
    out += ["", "(سطر 'نهاية الفترة' = صفقات لسه مفتوحة)"]
    tests = [r[2] for r in ok if r[2]]
    beat = sum(1 for m in tests if m["cagr"] > m["mkt_cagr"])
    out += ["", f"من كل الـ{len(tests)} إعداد (بالتكلفة الحقيقية): {beat} غلبوا السوق في الاختبار "
            f"({beat / max(len(tests), 1) * 100:.0f}%)، ومتوسط فرق العائد السنوي "
            f"{np.mean([m['cagr'] - m['mkt_cagr'] for m in tests]):+.1f}%.", ""]
    text = "\n".join(out)
    os.makedirs(bt.RES_DIR, exist_ok=True)
    tag = "latest" if a.source == "yfinance" else a.source
    with open(os.path.join(bt.RES_DIR, f"report_swing_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    if a.source == "yfinance":
        with open(BEST_FILE, "w", encoding="utf-8") as fh:
            json.dump({"params": best_p, "scores": use_sec, "sector": use_sec == "sec", "dev": best_dev, "test": best_test,
                       "live_extra": live_best, "old_real_test": old_real[1], "old_flat_test": old_flat[1],
                       "made": f"{datetime.now():%Y-%m-%d %H:%M}", "data_to": str(C.index[-1].date())},
                      fh, ensure_ascii=False, indent=1)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic-null", "synthetic-planted"])
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--base-bps", type=float, default=15.0, help="عمولة ومصاريف في كل اتجاه (من غير فرق السعر)")
    ap.add_argument("--min-turnover", type=float, default=1_000_000)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--block", type=int, default=63)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    try:
        run(a)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(bt.RES_DIR, exist_ok=True)
        with open(os.path.join(bt.RES_DIR, "error_swing_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
