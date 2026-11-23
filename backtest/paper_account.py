"""
الحساب التجريبي بالفلوس: 100,000 جنيه، أسهم بعدد صحيح، عمولة في كل عملية.

القرارات نفسها (إمتى ندخل/نخرج) من swing_core.decide - نفس القواعد اللي اتختبرت. هنا بس التنفيذ بالجنيه:
  - المسا (signals_report): بيقرر أوامر بكره (شراء بمبلغ / بيع كل الكمية) وبيحفظها في orders.
  - الصبح أثناء الجلسة (movers_live → live_tick): بينفذ الأوامر بسعر الافتتاح الحقيقي، وبيراقب كل سهم في
    المحفظة كل دورتين ونص: لو السعر كسر الوقف المتحرك = انعكاس → بيبيع فورًا وبيبعت تنبيه 🚨.
  - لو التشغيل اللحظي ماشتغلش، المسا بينفذ اللي فاضل ببيانات اليوم (الافتتاح، وأقل سعر للوقف).

signals/account.json:
  cash, capital, fee, start, last_date
  pos:    {اسم: {shares, entry_px, entry_date, cost (المدفوع بالعمولة), peak_px, last_px, atr, days}}
  orders: [{side: buy|sell, name, why, decided, budget (للشراء), ref_px}]
  history:[{date, time, side, name, shares, price, amount, fee, pnl, pnl_pct, why}]
  equity: [[تاريخ, قيمة الحساب, مؤشر السوق]]
"""
import json
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILE = os.path.join(ROOT, "signals", "account.json")
CAPITAL = float(os.environ.get("PAPER_CAPITAL", "100000"))
TZ = ZoneInfo("Africa/Cairo")
LIMIT_GAP = 0.09
REASON_AR = {"stop": "ضرب الوقف المتحرك", "stop_live": "انعكاس لحظي: كسر الوقف المتحرك",
             "rank": "ترتيبه في الموديل نزل", "time": "خلص مدة الاحتفاظ", "regime": "السوق قلب لاتجاه هابط"}


def money(x):
    return f"{x:,.0f}"


def px(x):
    return f"{x:.3f}" if x < 1 else f"{x:.2f}"


def new(fee, start):
    return {"capital": CAPITAL, "cash": CAPITAL, "fee": fee, "start": start, "last_date": None,
            "pos": {}, "orders": [], "history": [], "equity": [], "alerts": {}}


def load():
    try:
        with open(FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def save(acc):
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    acc["history"] = acc["history"][-1000:]
    acc["equity"] = acc["equity"][-2000:]
    with open(FILE, "w", encoding="utf-8") as fh:
        json.dump(acc, fh, ensure_ascii=False, indent=1)


def git_sync(msg="paper account [skip ci]"):
    """على GitHub Actions: commit + push لملف الحساب فورًا (عشان أي تنفيذ لحظي ميضيعش لو الـjob وقع، و"محفظتي" تشوفه)."""
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    import subprocess
    import random
    import time as _t
    run = lambda *a: subprocess.run(list(a), cwd=ROOT, capture_output=True, text=True, timeout=120)  # noqa: E731
    run("git", "add", FILE)
    if run("git", "diff", "--cached", "--quiet").returncode == 0:
        return
    run("git", "commit", "-m", msg)
    ref = os.environ.get("GITHUB_REF_NAME", "main") or "main"
    for _ in range(5):
        run("git", "pull", "--rebase", "--autostash", "-X", "theirs", "origin", ref)
        if run("git", "push").returncode == 0:
            return
        _t.sleep(3 + random.random() * 5)
    print("[الحساب التجريبي] push فشل، هيتعمل في آخر الـjob")


def stop_px(ps, k):
    return ps["peak_px"] * (1 - k * ps["atr"]) if k else None


def equity(acc, prices=None):
    prices = prices or {}
    return acc["cash"] + sum(ps["shares"] * prices.get(n, ps["last_px"]) for n, ps in acc["pos"].items())


# ============================================================
# التنفيذ
# ============================================================
def buy(acc, name, price, budget, when, why="signal", atr=0.03):
    """بيشتري أكبر عدد أسهم صحيح في حدود budget (والكاش). بيرجّع سجل العملية أو None."""
    fee = acc["fee"]
    budget = min(budget, acc["cash"])
    shares = int(math.floor(budget / (price * (1 + fee))))
    if shares <= 0 or name in acc["pos"]:
        return None
    amount = shares * price
    f = amount * fee
    acc["cash"] -= amount + f
    acc["pos"][name] = {"shares": shares, "entry_px": price, "entry_date": when[0], "cost": amount + f,
                        "peak_px": price, "last_px": price, "atr": float(atr), "days": 0, "last_date": ""}
    rec = {"date": when[0], "time": when[1], "side": "buy", "name": name, "shares": shares, "price": price,
           "amount": round(amount, 2), "fee": round(f, 2), "why": why}
    acc["history"].append(rec)
    return rec


def sell(acc, name, price, when, why):
    ps = acc["pos"].pop(name, None)
    if not ps:
        return None
    amount = ps["shares"] * price
    f = amount * acc["fee"]
    acc["cash"] += amount - f
    pnl = amount - f - ps["cost"]
    rec = {"date": when[0], "time": when[1], "side": "sell", "name": name, "shares": ps["shares"], "price": price,
           "amount": round(amount, 2), "fee": round(f, 2), "pnl": round(pnl, 2),
           "pnl_pct": round(pnl / ps["cost"] * 100, 2), "why": why, "entry_date": ps["entry_date"],
           "entry_px": ps["entry_px"], "days": ps["days"]}
    acc["history"].append(rec)
    return rec


def adjust_actions(acc, prevs, day, prev_day):
    """أسهم مجانية / تجزئة: TradingView بيعدّل قفلة امبارح (prev) بعد التوزيع، فلو prev بعيد عن آخر سعر عندنا
    بأكتر من 5% يبقى فيه إجراء على السهم → نضرب عدد الأسهم في المعامل ونقسم الأسعار عليه (القيمة متتغيرش).
    من غير ده، سهم وزّع مجاني 20% كان هيبان كأنه وقع 20% ويتباع غلط. بيرجّع قايمة رسايل."""
    out = []
    for n, ps in acc["pos"].items():
        pv = prevs.get(n)
        # بنقارن بس لو آخر سعر عندنا هو قفلة الجلسة اللي قبلها بالظبط (وإلا الفرق ممكن يكون حركة حقيقية)
        if (not pv or pv <= 0 or ps.get("adj_day") == day or ps.get("entry_date") == day
                or not prev_day or ps.get("last_date") != prev_day):
            continue
        f = ps["last_px"] / pv
        if 0.95 <= f <= 1.05 or not (0.2 <= f <= 10):
            continue
        old = ps["shares"]
        ps["shares"] = int(round(old * f))
        for k in ("entry_px", "peak_px", "last_px"):
            ps[k] = ps[k] / f
        ps["adj_day"] = day
        acc["history"].append({"date": day, "time": "", "side": "action", "name": n, "shares": ps["shares"],
                               "price": ps["last_px"], "amount": 0, "fee": 0, "why": f"تعديل {f:.4f}"})
        out.append(f"ℹ️ {n}: إجراء على السهم (أسهم مجانية/تجزئة بمعامل {f:.3f}) → الأسهم بقت {ps['shares']:,} بدل {old:,}، "
                   f"وسعر الشراء المعدّل {px(ps['entry_px'])}. قيمة المركز متغيرتش.")
    return out


def fill_orders(acc, opens, prevs, when):
    """تنفيذ أوامر الافتتاح: البيع الأول (عشان الكاش) وبعدين الشراء. opens/prevs: {اسم: سعر}.
    الأوامر اللي سعرها مش متاح بتفضل مستنية. بيرجّع قايمة العمليات."""
    done, left = [], []
    for o in sorted(acc["orders"], key=lambda o: o["side"] != "sell"):
        op = opens.get(o["name"])
        if not op or op <= 0:
            left.append(o)
            continue
        if o["side"] == "sell":
            ps0 = acc["pos"].get(o["name"], {})
            slip = o.get("slip", ps0.get("slip", 0.0))
            r = sell(acc, o["name"], round(op * (1 - slip), 4), when, o["why"])
        else:
            pv = prevs.get(o["name"])
            if pv and op >= pv * (1 + LIMIT_GAP):
                done.append({"side": "skip", "name": o["name"], "price": op,
                             "why": "فتح على الحد الأعلى، مش هنطارده"})
                continue
            slip = o.get("slip", 0.0)       # نص فرق سعر الشراء/البيع المقدّر: بنشتري أغلى شوية من الافتتاح زي الحقيقة
            r = buy(acc, o["name"], round(op * (1 + slip), 4), o["budget"], when, o.get("why", "signal"), o.get("atr", 0.03))
            if r:
                acc["pos"][o["name"]]["slip"] = slip
            if r is None:
                done.append({"side": "skip", "name": o["name"], "price": op, "why": "الكاش مش كفاية"})
                continue
        if r:
            done.append(r)
    acc["orders"] = left
    return done


def trade_line(r, day=None):
    pre = f"[{r['date']}] " if day and r.get("date") and r["date"] != day else ""
    return pre + _trade_line(r)


def _trade_line(r):
    if r["side"] == "note":
        return r["why"]
    if r["side"] == "buy":
        return (f"🛒 اشتريت {r['shares']:,} سهم {r['name']} بسعر {px(r['price'])} = {money(r['amount'])} جنيه "
                f"(+ عمولة {money(r['fee'])})")
    if r["side"] == "sell":
        icon = "💰" if r["pnl"] > 0 else "🔻"
        return (f"{icon} بعت {r['shares']:,} سهم {r['name']} بسعر {px(r['price'])} = {money(r['amount'])} جنيه | "
                f"{'مكسب' if r['pnl'] > 0 else 'خسارة'} {money(abs(r['pnl']))} جنيه ({r['pnl_pct']:+.1f}%) "
                f"بعد {r.get('days', 0)} جلسة | السبب: {REASON_AR.get(r['why'], r['why'])}")
    return f"⏭ {r['name']}: اتلغى الأمر ({r['why']})"


def summary_lines(acc, prices=None, mkt_ret=None):
    eq = equity(acc, prices)
    inv = eq - acc["cash"]
    pnl = eq - acc["capital"]
    closed = [h for h in acc["history"] if h["side"] == "sell"]
    wins = [h for h in closed if h["pnl"] > 0]
    L = [f"💼 الحساب التجريبي (بدأ {acc['start']} بـ {money(acc['capital'])} جنيه):",
         f"   القيمة دلوقتي {money(eq)} جنيه ({pnl / acc['capital'] * 100:+.2f}%) | كاش {money(acc['cash'])} | "
         f"في الأسهم {money(inv)}" + (f" | السوق في نفس الفترة {mkt_ret:+.2f}%" if mkt_ret is not None else "")]
    if closed:
        tot = sum(h["pnl"] for h in closed)
        L.append(f"   صفقات مقفولة {len(closed)}: كسبانة {len(wins)} | صافي المقفول {'+' if tot >= 0 else '-'}{money(abs(tot))} جنيه")
    return L


# ============================================================
# المراقبة اللحظية (بتتنادى من movers_live كل دورة)
# ============================================================
def live_params():
    """(معامل الوقف, عمق البيع اللحظي أو None) من إعدادات الباكتست. None = تنبيه لحظي بس والبيع بيستنى القفلة
    (الباكتست لقى إن البيع أول ما السعر يلمس الوقف أسوأ: الأسهم بتلمسه وترجع)."""
    try:
        with open(os.path.join(HERE, "results", "swing_best.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        k = float(m["params"].get("trail_k") or 0) or None
        extra = m.get("live_extra")
    except (OSError, ValueError, KeyError):
        k, extra = 2.5, None
    if os.environ.get("PAPER_INTRADAY_STOP", "true").lower() != "true":
        extra = None
    return k, extra


def live_tick(snap, now, kp, send, today=None):
    """snap: لقطة TradingView {اسم: {open, close, high, prev, date}}. بينفذ أوامر الافتتاح، وبيبيع فورًا لو
    سهم في المحفظة كسر الوقف المتحرك (انعكاس)، وبيحذّر مرة لو بينعكس بقوة وقرب من الوقف. بيرجّع True لو اتغير حاجة."""
    acc = load()
    if not acc:
        return False
    k, extra = kp if isinstance(kp, tuple) else (kp, None)
    today = today or now.strftime("%Y-%m-%d")
    when = (today, now.strftime("%H:%M"))
    changed = False
    live = {n: r for n, r in snap.items() if r.get("date") == today}
    if not live:
        return False
    # 0) أسهم مجانية/تجزئة قبل أي مقارنة بالوقف
    try:
        import egxdata
        prev_day = max([d for d in egxdata.load_bars() if d < today] or [None])
    except Exception:  # noqa: BLE001
        prev_day = None
    adj = adjust_actions(acc, {n: r.get("prev") for n, r in live.items()}, today, prev_day)
    if adj:
        changed = True
        send("\n".join(adj))
    # 1) أوامر الافتتاح
    if acc["orders"] and (now.hour, now.minute) >= (10, 2):
        opens = {n: (r.get("open") or None) for n, r in live.items()}
        prevs = {n: r.get("prev") for n, r in live.items()}
        done = fill_orders(acc, opens, prevs, when)
        if done:
            changed = True
            send("🧾 تنفيذ أوامر الافتتاح - " + f"{now:%H:%M}\n\n" + "\n".join(trade_line(r) for r in done)
                 + "\n\n" + "\n".join(summary_lines(acc, {n: r["close"] for n, r in live.items()})))
    # 2) الانعكاس: كسر الوقف أثناء الجلسة
    al = acc.setdefault("alerts", {})
    if al.get("date") != today:
        acc["alerts"] = al = {"date": today, "warned": []}
    sold = []
    for n in list(acc["pos"]):
        r = live.get(n)
        if not r:
            continue
        ps = acc["pos"][n]
        p = r["close"]
        stop = stop_px(ps, k)
        deep = stop_px(ps, k + extra) if (k and extra is not None) else None
        if deep and p <= deep:
            rec = sell(acc, n, round(p * (1 - ps.get("slip", 0.0)), 4), when, "stop_live")
            if rec:
                sold.append((rec, deep, r))
            continue
        broke = al.setdefault("broke", [])
        if stop and p <= stop and n not in broke:
            broke.append(n)
            changed = True
            send(f"🚨 انعكاس شديد في {n}: السعر {px(p)} كسر الوقف المتحرك {px(stop)} "
                 f"(النهاردة {((p / r['prev']) - 1) * 100 if r.get('prev') else 0:+.1f}%، أعلى سعر {px(r.get('high') or p)})\n"
                 f"المركز: {ps['shares']:,} سهم من {px(ps['entry_px'])} = {money(ps['shares'] * p)} جنيه "
                 f"({(p / ps['entry_px'] - 1) * 100:+.1f}%)\n"
                 + ("لو قفل تحت الوقف هبيعه عند افتتاح بكره (القاعدة المتختبرة)"
                    + (f"، ولو نزل لـ{px(deep)} هبيع فورًا." if deep else
                       ". الباكتست بيقول البيع أول ما يلمس الوقف أسوأ من استنى القفلة (أسهم كتير بتلمسه وترجع)، "
                       "بس لو عايز تخرج بإيدك دلوقتي ده وقته.")))
            continue
        hi = r.get("high") or p
        drop_hi = (p / hi - 1) if hi else 0
        if (stop and n not in al["warned"] and n not in broke and drop_hi <= -1.5 * ps["atr"] and p <= stop * (1 + ps["atr"])):
            al["warned"].append(n)
            changed = True
            send(f"⚠️ انعكاس بيتكوّن في {n}: نزل {abs(drop_hi) * 100:.1f}% من أعلى سعر النهاردة ({px(hi)}) ووصل {px(p)}\n"
                 f"الوقف المتحرك {px(stop)} (فاضل {(p / stop - 1) * 100:.1f}%). لو كسره هبيع على طول.\n"
                 f"المركز: {ps['shares']:,} سهم من {px(ps['entry_px'])} ({(p / ps['entry_px'] - 1) * 100:+.1f}%)")
    if sold:
        changed = True
        L = [f"🚨 انهيار - بيع فوري {now:%H:%M}", ""]
        for rec, stop, r in sold:
            L.append(trade_line(rec))
            L.append(f"   السعر نزل تحت مستوى الانهيار {px(stop)} (النهاردة {((r['close'] / r['prev']) - 1) * 100 if r.get('prev') else 0:+.1f}%، "
                     f"وأعلى سعر كان {px(r.get('high') or r['close'])}). الخروج بدري بيحمي من نزول أكتر.")
        L += ["", *summary_lines(acc, {n: r["close"] for n, r in live.items()}),
              "", "⚠️ تجريبي: نفس القاعدة اللي في الباكتست (الوقف المتحرك)، بس بتتنفذ لحظيًا. مش ضمان إنه هيكمل نزول."]
        send("\n".join(L))
    if changed:
        save(acc)
        git_sync()
    return changed
