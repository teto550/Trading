"""
محرك الدخول والخروج (Swing) - نفس القواعد بالظبط في الباكتست وفي التشغيل الحي.

الفكرة: الموديل (ensemble LightGBM على أهداف 20/40/60 يوم) بيدّي كل سهم درجة ترتيب 0..1.
القواعد بتحوّل الدرجة لقرارات صريحة:
  - دخول: السوق في اتجاه صاعد (مؤشر متوسط الأسهم فوق متوسطه المتحرك) + السهم من أعلى q_in
           + (اختياري) السهم فوق متوسط 50 يوم + مش قافل على الحد + فيه مكان في المحفظة.
           التنفيذ عند افتتاح الجلسة اللي بعد الإشارة (مش بسعر القفلة اللي فات).
  - خروج: وقف متحرك (trail_k × التذبذب اليومي تحت أعلى سعر من وقت الدخول)،
          أو الترتيب نزل تحت q_out، أو عدّى max_hold يوم، أو (اختياري) السوق قلب.
  - كل صفقة بتاخد نفس الوزن (رأس المال / max_pos)، وتكلفة `cost` في كل اتجاه.
"""
import numpy as np
import pandas as pd

MAX_ABS = 0.25      # أي حركة يومية أكبر من كده = تقسيم/بيانات غلط (بتتحسب صفر)
LIMIT_GAP = 0.09    # افتتاح فوق قفلة امبارح بـ9%+ = غالبًا مقفول على الحد، مش هتلحق تشتري

DEFAULT = dict(q_in=0.95, q_out=0.50, max_pos=8, trail_k=3.0, max_hold=40, min_hold=5,
               regime=100, trend50=True, exit_on_regime=False, cost=0.002)

REASON_AR = {"stop": "ضرب الوقف المتحرك", "rank": "ترتيبه في الموديل نزل", "time": "خلص مدة الاحتفاظ",
             "regime": "السوق قلب لاتجاه هابط", "end": "نهاية الفترة"}


def _clean(x):
    x = np.where(np.isfinite(x) & (np.abs(x) <= MAX_ABS), x, 0.0)
    return x


def prepare(C, H, L, V, O, S, liquid, lock):
    """بيجهّز المصفوفات اللي المحاكي محتاجها. C..O DataFrames بنفس الفهرس، S مصفوفة (T,N) درجات 0..1."""
    Cv = C.values.astype(float)
    Cf = C.ffill().values.astype(float)
    Ov = O.values.astype(float)
    prev = np.vstack([np.full((1, Cf.shape[1]), np.nan), Cf[:-1]])
    rc = _clean(Cf / prev - 1)                       # قفلة لقفلة (بالسعر المكمّل)
    oc = _clean(Cv / Ov - 1)                         # يوم الدخول: افتتاح → قفلة
    co = _clean(Ov / prev - 1)                       # يوم الخروج: قفلة امبارح → افتتاح
    gap = np.where(np.isfinite(Ov / prev), Ov / prev - 1, np.nan)
    pc = C.shift(1)
    tr = pd.concat([(H - L), (H - pc).abs(), (L - pc).abs()]).groupby(level=0).max().reindex(C.index)
    atr = (tr.rolling(14, min_periods=10).mean() / C).values
    atr = np.clip(np.nan_to_num(atr, nan=0.03), 0.005, 0.08)
    r = C / C.shift(1) - 1
    r = r.where(r.abs() <= MAX_ABS)
    P = np.exp(np.log1p(r).fillna(0).cumsum()).where(C.notna())
    sma50 = (P > P.rolling(50, min_periods=50).mean()).values
    rl = r.where(liquid)
    idx = (1 + rl.mean(axis=1).fillna(0)).cumprod()
    reg = {n: (idx > idx.rolling(n, min_periods=n).mean()).values for n in (50, 100, 200)}
    reg[0] = np.ones(len(idx), bool)
    ok_in = liquid.values & ~lock.values & np.isfinite(S) & np.isfinite(Cv)
    order = []
    for t in range(len(C)):
        m = ok_in[t] & (S[t] >= 0.80)
        ii = np.where(m)[0]
        order.append(ii[np.argsort(-S[t, ii])])
    return dict(dates=C.index, names=np.array(C.columns), C=Cv, O=Ov, rc=rc, oc=oc, co=co, gap=gap,
                atr=atr, sma50=sma50, S=S, ok_in=ok_in, reg=reg, order=order, mkt=rl.mean(axis=1).fillna(0).values)


def decide(D, p, t, pos, pending_exit):
    """قرارات قفلة اليوم t: (خروج {i: سبب}, دخول [i...]). pos: {i: dict(rel, peak, atr, days)}."""
    S = D["S"]
    reg = D["reg"][p["regime"]][t]
    exits = {}
    for i, ps in pos.items():
        if i in pending_exit:
            continue
        why = None
        if p["trail_k"] and ps["rel"] < ps["peak"] * (1 - p["trail_k"] * ps["atr"]):
            why = "stop"
        elif ps["days"] >= p["max_hold"]:
            why = "time"
        elif ps["days"] >= p["min_hold"] and np.isfinite(S[t, i]) and S[t, i] < p["q_out"]:
            why = "rank"
        elif p["exit_on_regime"] and not reg:
            why = "regime"
        if why:
            exits[i] = why
    entries = []
    if reg:
        free = p["max_pos"] - (len(pos) - len(exits) - len([i for i in pending_exit if i in pos]))
        if free > 0:
            for i in D["order"][t]:
                if S[t, i] < p["q_in"]:
                    break
                if i in pos or (p["trend50"] and not D["sma50"][t, i]):
                    continue
                entries.append(int(i))
                if len(entries) >= free:
                    break
    return exits, entries


def new_book():
    return {"cash": 1.0, "pos": {}, "pend_x": {}, "pend_in": [], "trades": []}


def exec_open(D, p, t, bk):
    """تنفيذ قرارات قفلة امبارح عند افتتاح اليوم t: خروج الأول وبعدين دخول. بيرجّع (خرج, دخل)."""
    cost, pos = p["cost"], bk["pos"]
    out_, in_ = [], []
    for i in list(bk["pend_x"]):
        if i not in pos:
            bk["pend_x"].pop(i)
            continue
        if not np.isfinite(D["O"][t, i]):
            continue                                       # السهم موقوف: نحاول بكره
        ps = pos.pop(i)
        g = 1 + (D["rc"][t, i] if p.get("exec_close") else D["co"][t, i])   # exec_close: تنفيذ بقفلة اليوم (اختبار تأخير)
        ps["val"] *= g
        ps["rel"] *= g
        bk["cash"] += ps["val"] * (1 - cost)
        tr = {"i": i, "et": ps["et"], "xt": t, "ret": ps["rel"] * (1 - cost) ** 2 - 1,
              "days": ps["days"], "why": bk["pend_x"].pop(i)}
        bk["trades"].append(tr)
        out_.append(tr)
    eq_now = bk["cash"] + sum(ps["val"] for ps in pos.values())
    slot = eq_now / p["max_pos"]
    for i in bk["pend_in"]:
        if i in pos or len(pos) >= p["max_pos"]:
            continue
        if not np.isfinite(D["O"][t, i]) or not (D["gap"][t, i] < LIMIT_GAP):
            continue
        alloc = min(bk["cash"], slot)
        if alloc < 0.25 * slot:
            break
        bk["cash"] -= alloc
        g = 1.0 if p.get("exec_close") else 1 + D["oc"][t, i]
        pos[i] = {"val": alloc * (1 - cost) * g, "rel": g, "peak": max(g, 1.0), "atr": float(D["atr"][t - 1, i]),
                  "days": 0, "et": t}
        in_.append(i)
    bk["pend_in"] = []
    return out_, in_


def mark_close(D, t, bk):
    for i, ps in bk["pos"].items():
        if ps["et"] != t:
            g = 1 + D["rc"][t, i]
            ps["val"] *= g
            ps["rel"] *= g
            ps["days"] += 1
            ps["peak"] = max(ps["peak"], ps["rel"])
    inv = sum(ps["val"] for ps in bk["pos"].values())
    eq = bk["cash"] + inv
    return eq, (inv / eq if eq > 0 else 0.0)


def simulate(D, p, t0, t1):
    """محاكاة يوم بيوم من t0 لـ t1. بيرجّع (equity يومي, exposure يومي, صفقات)."""
    T = len(D["dates"])
    t1 = min(t1, T)
    bk = new_book()
    eq = np.full(T, np.nan)
    expo = np.zeros(T)
    for t in range(t0, t1):
        exec_open(D, p, t, bk)
        eq[t], expo[t] = mark_close(D, t, bk)
        if t < t1 - 1:
            ex, en = decide(D, p, t, bk["pos"], bk["pend_x"])
            bk["pend_x"].update(ex)
            bk["pend_in"] = en
    trades = list(bk["trades"])
    for i, ps in bk["pos"].items():                        # تقفيل دفتري في الآخر
        trades.append({"i": i, "et": ps["et"], "xt": t1 - 1, "ret": ps["rel"] * (1 - p["cost"]) ** 2 - 1,
                       "days": ps["days"], "why": "end", "open": True})
    return eq, expo, trades


def metrics(D, eq, expo, trades, a, b):
    """مقاييس الفترة [a, b) بالتواريخ (Timestamp)."""
    dates = D["dates"]
    m = (dates >= a) & (dates < b) & np.isfinite(eq)
    if m.sum() < 60:
        return None
    e = eq[m] / eq[m][0]
    rets = np.diff(e) / e[:-1]
    yrs = m.sum() / 245.0
    cagr = e[-1] ** (1 / yrs) - 1
    vol = rets.std() * np.sqrt(245)
    peak = np.maximum.accumulate(e)
    mdd = (e / peak - 1).min()
    mk = D["mkt"][m]
    mk_e = np.cumprod(1 + mk)
    mk_cagr = mk_e[-1] ** (1 / yrs) - 1
    mk_mdd = (mk_e / np.maximum.accumulate(mk_e) - 1).min()
    ti = [x for x in trades if a <= dates[x["et"]] < b]
    r = np.array([x["ret"] for x in ti]) if ti else np.array([])
    wins, loss = r[r > 0], r[r <= 0]
    return {"cagr": cagr * 100, "vol": vol * 100, "sharpe": (rets.mean() * 245) / (vol + 1e-12), "mdd": mdd * 100,
            "calmar": cagr / (abs(mdd) + 1e-9), "total": (e[-1] - 1) * 100,
            "mkt_cagr": mk_cagr * 100, "mkt_mdd": mk_mdd * 100, "mkt_total": (mk_e[-1] - 1) * 100,
            "n": len(r), "win": float((r > 0).mean() * 100) if len(r) else float("nan"),
            "avg": float(r.mean() * 100) if len(r) else float("nan"),
            "avg_win": float(wins.mean() * 100) if len(wins) else float("nan"),
            "avg_loss": float(loss.mean() * 100) if len(loss) else float("nan"),
            "pf": float(wins.sum() / (-loss.sum() + 1e-12)) if len(loss) else float("nan"),
            "hold": float(np.mean([x["days"] for x in ti])) if ti else float("nan"),
            "expo": float(expo[m].mean() * 100)}


def yearly(D, eq, a):
    s = pd.Series(eq, index=D["dates"]).dropna()
    s = s[s.index >= a]
    mk = pd.Series(np.cumprod(1 + D["mkt"]), index=D["dates"])[s.index]
    out = []
    for y, g in s.groupby(s.index.year):
        mg = mk[g.index]
        prev_s = s[s.index < g.index[0]]
        prev_m = mk[mk.index < g.index[0]]
        b0 = prev_s.iloc[-1] if len(prev_s) else g.iloc[0]
        m0 = prev_m.iloc[-1] if len(prev_m) else mg.iloc[0]
        out.append((y, (g.iloc[-1] / b0 - 1) * 100, (mg.iloc[-1] / m0 - 1) * 100))
    return out
