#!/usr/bin/env python3
"""
================================================================
   EGX Paper Trade - تجربة حية من غير فلوس لاختيارات XGBoost (احتفاظ 20 يوم)
================================================================
الباكتست وصل لاستراتيجية واحدة بس عدّت الحكم: XGBoost باحتفاظ 20 يوم، بميزة صغيرة (حوالي +0.3%
للصفقة بعد التكلفة) وغير ثابتة في آخر 3 سنين. الباكتست لوحده مش كفاية، فالسكريبت ده بيعمل تجربة
حية: كل يوم تداول بعد القفل بيسجّل "دفعة" اختيارات، وبعد 20 يوم تداول بيحسب نتيجتها الفعلية
(شراء عند افتتاح اليوم التالي، بيع عند قفلة اليوم العشرين) ويقارنها بمتوسط السوق وبـ low-vol.

كل دفعة بتسجّل:
  - universe: كل الأسهم الصالحة (سيولة كفاية + مش قافلة على الحد) = "السوق" اللي بنقارن بيه
  - xgb: أعلى 20% حسب XGBoost (مدرّب على كل التاريخ اللي نتيجته خلصت)
  - lowvol: أقل 20% تذبذب (مرجع: ممكن يطلع إن XGBoost مجرد low-vol)

⚠️ ميزة +0.3% للصفقة صغيرة جدًا قدام تذبذب الصفقات؛ محتاجين شهور (مش أسابيع) عشان نقول
لو الميزة حقيقية. الدفعات المتجاورة بتتشارك أيام فمتتحسبش كعينات مستقلة.

الاستخدام:
  python backtest/paper_trade.py                 # تشغيل عادي (GitHub Actions بعد القفل)
  python backtest/paper_trade.py --no-telegram
================================================================
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import backtest as bt  # noqa: E402
import backtest_horizons as bh  # noqa: E402

PAPER_DIR = os.path.join(ROOT, "paper")
HOLD = 20
EXPECTED_NET_PCT = 0.28   # من الباكتست: +1.4 bps/يوم × 20 يوم، بعد التكلفة (2022+)


def prepare(panel, min_turnover):
    C, H, L, V, O = bt.clean_panel(panel)
    f, mkt, _n, _noc, lock, liquid, flat59 = bt.build_features(C, H, L, V, O, min_turnover)
    lr = np.log1p(f["r1"])
    f["r60"] = np.expm1(lr.rolling(60, min_periods=60).sum())
    f["r120"] = np.expm1(lr.rolling(120, min_periods=120).sum())
    N = C.shape[1]
    cs = {k: bt.csrank(f[k], liquid).values for k in bh.XF2}
    Xcs = np.stack([cs[k] for k in bh.XF2] +
                   [np.repeat(mkt[k].values[:, None], N, axis=1) for k in bt.MF], axis=2).astype(np.float32)
    feat_ok = np.isfinite(Xcs).all(axis=2) & liquid.values & ~lock.values
    return {"C": C, "O": O, "f": f, "Xcs": Xcs, "feat_ok": feat_ok, "TGT": bh.build_target(C, O, HOLD)}


def data_is_complete(C):
    """آخر يوم في البيانات لازم يكون فيه تقريبًا نفس عدد الأسهم المعتاد، وإلا Yahoo لسه ماحدّثش."""
    if len(C) < 30:
        return False
    cnt = int(C.iloc[-1].notna().sum())
    med = float(C.iloc[-21:-1].notna().sum(axis=1).median())
    return cnt >= 0.7 * med


def make_cohort(P, trees):
    import xgboost as xgb
    T = P["C"].shape[0]
    E = P["feat_ok"] & np.isfinite(P["TGT"].values)
    if int(E.sum()) < 5000:
        raise bt.BacktestError(f"بيانات التدريب قليلة ({int(E.sum())} صف)")
    Y = (pd.DataFrame(np.where(E, P["TGT"].values, np.nan)).rank(axis=1, pct=True) - 0.5).values
    model = xgb.XGBRegressor(
        n_estimators=trees, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
        min_child_weight=50, reg_lambda=10.0, tree_method="hist", n_jobs=2, random_state=0)
    model.fit(P["Xcs"][E], Y[E].astype(np.float32))
    t = T - 1
    cols = np.where(P["feat_ok"][t])[0]
    if len(cols) < 40:
        raise bt.BacktestError(f"عدد الأسهم الصالحة النهاردة قليل ({len(cols)})")
    score = model.predict(P["Xcs"][t, cols])
    k = max(len(cols) // 5, 5)
    names = np.array(P["C"].columns)
    vol = P["f"]["vol20"].values[t, cols]
    return {
        "date": str(P["C"].index[t].date()),
        "universe": sorted(names[cols].tolist()),
        "xgb": names[cols[np.argsort(-score)[:k]]].tolist(),
        "lowvol": names[cols[np.argsort(vol)[:k]]].tolist(),
    }


def score_cohorts(P, state, cost):
    """نتيجة كل دفعة عدّى عليها 20 يوم تداول: الفرق عن متوسط السوق (universe) في نفس الفترة."""
    pos = {str(d.date()): i for i, d in enumerate(P["C"].index)}
    rows = []
    for c in state["cohorts"]:
        i = pos.get(c["date"])
        if i is None:
            continue
        row = P["TGT"].iloc[i]
        mk = row.reindex(c["universe"]).dropna()
        if len(mk) < 20:
            continue                                       # لسه مخلصتش (أو بيانات ناقصة)
        rec = {"date": c["date"], "mkt": float(mk.mean()), "n_univ": int(len(mk))}
        for key in ("xgb", "lowvol"):
            v = row.reindex(c[key]).dropna()
            rec[key] = float(v.mean() - mk.mean()) if len(v) >= 3 else float("nan")
        rows.append(rec)
    return rows


def summarize(rows, cost):
    rt = 2 * cost
    if not rows:
        return None
    d = pd.DataFrame(rows)
    out = {"n": len(d), "first": d["date"].iloc[0], "last": d["date"].iloc[-1],
           "mkt_pct": float(d["mkt"].mean() * 100)}
    for key in ("xgb", "lowvol"):
        ex = d[key].dropna()
        out[key] = {"ex_pct": float(ex.mean() * 100), "net_pct": float((ex.mean() - rt) * 100),
                    "hit": float((ex > rt).mean()), "n": int(len(ex)),
                    "std_pct": float(ex.std(ddof=1) * 100) if len(ex) > 2 else float("nan")}
    return out


def fmt_summary_md(state, stats, new_cohort, cost_bps):
    L = [f"# EGX Paper Trade - آخر تحديث {datetime.now():%Y-%m-%d %H:%M}", "",
         f"- احتفاظ {HOLD} يوم تداول، شراء عند افتتاح اليوم التالي للإشارة، تكلفة {cost_bps:.1f} bps في الاتجاه",
         f"- دفعات متسجلة: {len(state['cohorts'])} | ناضجة (عدّى عليها {HOLD} يوم): {stats['n'] if stats else 0}",
         f"- توقع الباكتست: حوالي {EXPECTED_NET_PCT:+.2f}% للصفقة بعد التكلفة (ميزة صغيرة وغير ثابتة)", ""]
    if stats:
        L += ["| | الفرق عن السوق | بعد التكلفة | نسبة الدفعات الكسبانة | تذبذب الدفعة |", "|---|---|---|---|---|"]
        for key, label in (("xgb", "XGBoost"), ("lowvol", "Low-vol (مرجع)")):
            s = stats[key]
            L.append(f"| {label} | {s['ex_pct']:+.2f}% | {s['net_pct']:+.2f}% | {s['hit'] * 100:.0f}% "
                     f"({s['n']}) | {s['std_pct']:.2f}% |")
        L += ["", f"متوسط عائد السوق (الأسهم الصالحة) لنفس الفترات: {stats['mkt_pct']:+.2f}% للصفقة",
              f"الدفعات من {stats['first']} إلى {stats['last']}"]
    L += ["", f"## اختيارات آخر دفعة ({new_cohort['date']})",
          f"XGBoost ({len(new_cohort['xgb'])}): " + ", ".join(t.replace(".CA", "") for t in new_cohort["xgb"]),
          "", f"Low-vol ({len(new_cohort['lowvol'])}): " + ", ".join(t.replace(".CA", "") for t in new_cohort["lowvol"])]
    return "\n".join(L)


def tg_message(new_cohort, stats, n_total, cost_bps, created, stale_days=0):
    lines = [f"📒 Paper-trade (من غير فلوس) - آخر دفعة {new_cohort['date']}"]
    lines.append("ℹ️ Yahoo بيتأخر عن السوق المصري جلسة أو أكتر، فتاريخ الدفعة هو آخر جلسة عنده. "
                 "الاختيارات دي للقياس بعد 20 يوم، مش للشراء.")
    if created:
        names = ", ".join(t.replace(".CA", "") for t in new_cohort["xgb"][:30])
        lines.append(f"دفعة جديدة: {len(new_cohort['xgb'])} سهم من {len(new_cohort['universe'])} صالح (XGBoost، احتفاظ {HOLD} يوم)")
        lines.append(names)
    else:
        lines.append("الدفعة دي متسجلة قبل كده، مفيش دفعة جديدة.")
    lines.append("")
    if stats:
        x, lv = stats["xgb"], stats["lowvol"]
        lines.append(f"📊 {stats['n']} دفعة ناضجة من {n_total}:")
        lines.append(f"XGBoost: {x['ex_pct']:+.2f}% عن السوق للصفقة، بعد التكلفة {x['net_pct']:+.2f}% "
                     f"(كسب {x['hit'] * 100:.0f}%)")
        lines.append(f"Low-vol: {lv['ex_pct']:+.2f}%، بعد التكلفة {lv['net_pct']:+.2f}%")
        lines.append(f"توقع الباكتست حوالي {EXPECTED_NET_PCT:+.2f}%. العينة صغيرة، متحكمش قبل شهور.")
    else:
        lines.append(f"لسه مفيش دفعة عدّى عليها {HOLD} يوم تداول ({n_total} دفعة متسجلة). "
                     f"أول نتيجة بعد حوالي 4 أسابيع.")
    return "\n".join(lines)


def process(panel, state, a, send=True):
    """بيتنفذ مرة في اليوم: بيسجل دفعة النهاردة (لو البيانات كاملة ومتسجلتش قبل كده) وبيحسب النتايج."""
    cost = a.cost_bps / 1e4
    P = prepare(panel, a.min_turnover)
    last = str(P["C"].index[-1].date())
    existing = {c["date"]: c for c in state["cohorts"]}
    created = False
    if last in existing:
        cohort = existing[last]
    elif not data_is_complete(P["C"]):
        raise bt.BacktestError(f"بيانات {last} ناقصة (Yahoo لسه ماحدّثش). هيتعاد في التشغيل الجاي.")
    else:
        cohort = make_cohort(P, a.trees)
        state["cohorts"].append(cohort)
        created = True
    rows = score_cohorts(P, state, cost)
    stats = summarize(rows, cost)
    state["last_stats"] = stats
    state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return cohort, stats, created


def run(a):
    os.makedirs(PAPER_DIR, exist_ok=True)
    picks_file = os.path.join(PAPER_DIR, "picks.json")
    state = {"cohorts": []}
    if os.path.exists(picks_file):
        with open(picks_file, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    panel = bt.fetch_yfinance(bt.load_tickers())
    cohort, stats, created = process(panel, state, a)
    # الرسالة مرة في الأسبوع (الخميس) + أول دفعة + أول مرة دفعة تنضج، عشان مانزعجكش كل يوم
    cairo = datetime.now(ZoneInfo("Africa/Cairo"))
    n_mat = stats["n"] if stats else 0
    send = ((created and len(state["cohorts"]) == 1) or cairo.weekday() == 3
            or (n_mat > 0 and state.get("notified_matured", 0) == 0))
    state["notified_matured"] = max(state.get("notified_matured", 0), n_mat)
    stale = (cairo.date() - datetime.strptime(cohort["date"], "%Y-%m-%d").date()).days
    with open(picks_file, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)
    with open(os.path.join(PAPER_DIR, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(fmt_summary_md(state, stats, cohort, a.cost_bps))
    print(f"[done] دفعة {cohort['date']} ({'جديدة' if created else 'موجودة'}) | ناضجة: {n_mat} | "
          f"تأخر Yahoo: {stale} يوم | رسالة تليجرام: {'أيوه' if send else 'لأ'}")
    if send and not a.no_telegram:
        bt.send_telegram(tg_message(cohort, stats, len(state["cohorts"]), a.cost_bps, created, stale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=17.5)
    ap.add_argument("--min-turnover", type=float, default=1_000_000)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()
    try:
        run(a)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)
        os.makedirs(PAPER_DIR, exist_ok=True)
        with open(os.path.join(PAPER_DIR, "error_latest.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M}\n{tb}")
        if not a.no_telegram:
            kind = "فشل (متوقع)" if isinstance(e, bt.BacktestError) else "خطأ في الكود"
            bt.send_telegram(f"📒❌ Paper-trade وقع - {kind}:\n{type(e).__name__}: {str(e)[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
