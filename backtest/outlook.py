#!/usr/bin/env python3
"""
مرجع الدخول: لكل سهم مرشح (طلوع أو نزول) على فترات 20/40/60 يوم:
  - المتوسط المتوقع للعائد + مدى (من 20% لـ80% احتمال) + السعر المقابل
  - الفترة الأنسب
  - الأسهم الطالعة حاليًا ومتوقع تعكس اتجاهها
بيتسجل كل توقع ويتقيّم لما الفترة تخلص (نسبة وقوعه جوه المدى والاتجاه الصح).
الموديل: LightGBM quantile (20/50/80%) على نفس مدخلات Alpha158، وبيتقاس تغطيته على آخر 250 يوم قبل ما يتعرض.
"""
import json
import os

import numpy as np

import backtest as bt
import backtest_horizons as bh
import direction as dr

HS = (20, 40, 60)
QS = (0.2, 0.5, 0.8)
COST = 0.004          # تكلفتك الفعلية للصفقة (~40 bps)
TOP = 10
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "horizon")
LOG = os.path.join(OUT_DIR, "outlook_log.json")


def _mk(q, trees=150):
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=trees, max_depth=3, num_leaves=8,
                             learning_rate=0.05, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                             min_child_samples=100, reg_lambda=10.0, n_jobs=2, random_state=0, verbose=-1)


def _fit_q(X, y, w, Xnew):
    out = []
    for q in QS:
        m = _mk(q)
        m.fit(X, y, sample_weight=w)
        out.append(m.predict(Xnew))
    return np.sort(np.stack(out, axis=1), axis=1)      # (n,3) مرتبة (q20<=q50<=q80)


def predict_all(P, hs=HS):
    """بيرجّع {h: {"cols", "q": (n,3), "cov": تغطية آخر 250 يوم}} لآخر يوم."""
    C = P["C"]
    t = C.shape[0] - 1
    cols = np.where(P["feat_ok"][t])[0]
    res = {}
    for h in hs:
        tgt = bh.build_target(C, P["O"], h).values
        E = P["feat_ok"] & np.isfinite(tgt)
        if int(E.sum()) < 5000:
            continue
        y = np.clip(tgt, -0.5, 1.0)
        q = _fit_q(P["Xcs"][E], y[E], dr.recency_w(E), P["Xcs"][t, cols])
        cov = None
        try:      # تغطية خارج التدريب: آخر 250 يوم (التدريب قبلها بـh يوم)
            T = C.shape[0]
            s0 = T - 250
            Etr = E.copy()
            Etr[s0 - h:] = False
            Ete = np.zeros_like(E)
            Ete[s0 - h:T - h] = E[s0 - h:T - h]       # بس الأيام اللي نتيجتها خلصت
            Ete[:s0] = False
            if Etr.sum() > 5000 and Ete.sum() > 500:
                qt = _fit_q(P["Xcs"][Etr], y[Etr], dr.recency_w(Etr), P["Xcs"][Ete])
                yt = y[Ete]
                cov = {"below20": float((yt < qt[:, 0]).mean() * 100), "below80": float((yt < qt[:, 2]).mean() * 100),
                       "inside": float(((yt >= qt[:, 0]) & (yt <= qt[:, 2])).mean() * 100),
                       "up_when_med_pos": float((yt[qt[:, 1] > COST] > 0).mean() * 100)
                       if (qt[:, 1] > COST).sum() > 50 else None}
        except Exception as e:  # noqa: BLE001
            print(f"[outlook] تغطية {h} فشلت: {str(e)[:100]}")
        res[h] = {"cols": cols, "q": q, "cov": cov}
        print(f"[outlook] {h} يوم تم" + (f" | تغطية {cov['inside']:.0f}% (المفروض 60%)" if cov else ""))
    return res


def build(P, res_rank, outl):
    """res_rank: {h: stocks dict من top_picks} لترتيب الأسهم. بيرجّع (نص الرسالة, سجل للتتبع)."""
    C = P["C"]
    t = C.shape[0] - 1
    hs = [h for h in HS if h in outl and h in res_rank]
    if len(hs) < 2:
        return None, None
    cols = outl[hs[0]]["cols"]
    names = [C.columns[c].replace(".CA", "") for c in cols]
    n = len(cols)
    mr = np.zeros(n)
    for h in hs:
        rk = res_rank[h]["stocks"]
        mr += np.array([rk.get(nm, {"rank": n})["rank"] for nm in names], dtype=float)
    mr /= len(hs)
    r20 = (C.iloc[t, cols].values / C.iloc[t - 20, cols].values - 1)

    def row(i, side):
        col, last = int(cols[i]), float(C.iloc[t, cols[i]])
        parts, best, bscore = [], None, -9
        for h in hs:
            q20, q50, q80 = outl[h]["q"][i]
            parts.append(f"{h}ي: {q50 * 100:+.0f}% ({q20 * 100:+.0f}..{q80 * 100:+.0f}) ← {last * (1 + q50):.2f}")
            sc = side * q50 / max(q80 - q20, 1e-6)
            if sc > bscore:
                bscore, best = sc, h
        lv = None
        try:
            import tomorrow_report as tr
            lv = tr.levels(C, P["H"], P["L"], col, t)
        except Exception:  # noqa: BLE001
            pass
        stop = f" | وقف {lv['stop']:.2f}" if (lv and side > 0) else ""
        return (f"• {names[i]} (حاليًا {last:.2f}, آخر 20 يوم {r20[i] * 100:+.0f}%): " + " | ".join(parts)
                + f"{stop} | الأنسب: {best} يوم"), best

    ups = [i for i in np.argsort(mr) if all(outl[h]["q"][i][1] > COST for h in hs)][:TOP]
    downs = [i for i in np.argsort(-mr) if all(outl[h]["q"][i][1] < -COST for h in hs)][:TOP]
    rev = [i for i in downs if r20[i] > 0.03][:5]
    log = {"date": str(C.index[-1].date()), "picks": []}
    L = [f"📘 مرجع الدخول - {C.index[-1].date()}",
         "المتوقع = وسيط العائد، والقوسين المدى (20%..80% احتمال)، والسعر المقابل للوسيط. العائد من افتتاح الجلسة الجاية.", ""]
    for title, idx, side in (("🟢 فرص طلوع (ترتيب موديلات 20/40/60 يوم):", ups, 1),
                             ("🔴 متوقع تنزل:", [i for i in downs if i not in rev], -1),
                             ("🔄 كانت طالعة ومتوقع تعكس وتنزل:", rev, -1)):
        if not idx:
            continue
        L += [title]
        for i in idx:
            txt, best = row(i, side)
            L.append(txt)
            log["picks"].append({"n": names[i], "side": side, "last": float(C.iloc[t, cols[i]]),
                                 "q": {str(h): [float(x) for x in outl[h]["q"][i]] for h in hs}})
        L.append("")
    if len(L) <= 3:
        L.append("مفيش أسهم عدّت الشرط النهاردة (وسيط العائد المتوقع أكبر من تكلفتك 0.4% في الفترات التلاتة).")
    cov = [f"{h}ي: {outl[h]['cov']['inside']:.0f}%" for h in hs if outl[h].get("cov")]
    L += ["دقة المدى: في آخر 250 يوم المدى ده غطى " + (" | ".join(cov) if cov else "(مش متاح)") +
          " من النتايج الفعلية (المفروض 60%) - لو أقل من كده يبقى المدى أضيق من الواقع.",
          "⚠️ ده تقدير إحصائي مش وعد. في الباكتست الفترات دي (خصوصًا 40 و60 يوم) هي الأقوى، لكن الأرقام ليها عدد فترات مستقلة "
          "قليل، والنزول بيتنفذ بس لو عندك طريقة بيع فعلية. بسجّل كل توقع وهقيّمه لما الفترة تخلص."]
    return "\n".join(L), log


def log_and_score(C, O, log):
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        with open(LOG, encoding="utf-8") as f:
            hist = json.load(f)
    except Exception:  # noqa: BLE001
        hist = []
    msgs = []
    names = [c.replace(".CA", "") for c in C.columns]
    dates = [str(d.date()) for d in C.index]
    for rec in hist:
        if rec.get("scored") or rec["date"] not in dates:
            continue
        t0 = dates.index(rec["date"])
        res = {}
        for h in HS:
            if t0 + h >= len(dates):
                continue
            rows = []
            for p in rec["picks"]:
                if p["n"] not in names or str(h) not in p["q"]:
                    continue
                j = names.index(p["n"])
                o1 = float(O.iloc[t0 + 1, j]) if t0 + 1 < len(O) else None
                if not o1 or not np.isfinite(o1):
                    continue
                r = float(C.iloc[t0 + h, j] / o1 - 1)
                q20, q50, q80 = p["q"][str(h)]
                rows.append({"side": p["side"], "r": r, "q50": q50, "in": q20 <= r <= q80,
                             "dir": (r > 0) == (p["side"] > 0)})
            if rows:
                res[str(h)] = rows
        rec.setdefault("done", {}).update({k: v for k, v in res.items() if k not in rec.get("done", {})})
        if all(str(h) in rec.get("done", {}) for h in HS):
            rec["scored"] = True
        for h, rows in res.items():
            if rec.get("sent", {}).get(h):
                continue
            rec.setdefault("sent", {})[h] = True
            n = len(rows)
            up = [x for x in rows if x["side"] > 0]
            dn = [x for x in rows if x["side"] < 0]
            line = [f"🏁 نتيجة مرجع الدخول بتاع {rec['date']} بعد {h} يوم:",
                    f"المدى غطى {sum(x['in'] for x in rows) / n * 100:.0f}% من النتايج (المفروض 60%) | الاتجاه صح {sum(x['dir'] for x in rows) / n * 100:.0f}%"]
            if up:
                line.append(f"الطالعة ({len(up)}): متوسط العائد الفعلي {np.mean([x['r'] for x in up]) * 100:+.1f}% "
                            f"مقابل متوقع {np.mean([x['q50'] for x in up]) * 100:+.1f}%")
            if dn:
                line.append(f"النازلة ({len(dn)}): متوسط العائد الفعلي {np.mean([x['r'] for x in dn]) * 100:+.1f}% "
                            f"مقابل متوقع {np.mean([x['q50'] for x in dn]) * 100:+.1f}%")
            msgs.append("\n".join(line))
    if log and not any(r["date"] == log["date"] for r in hist):
        hist.append(log)
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump(hist[-60:], f, ensure_ascii=False)
    return msgs
