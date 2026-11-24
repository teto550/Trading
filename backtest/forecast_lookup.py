"""
رد البوت على اسم سهم: تحليل + توقع الجلسة الجاية و5 أيام و10 أيام (شراء/بيع) + أهم الأخبار.

مبيدرّبش موديل وقت السؤال (ده بياخد دقايق): التقارير اليومية (tomorrow_report و horizon_report)
بتحفظ توقع كل الأسهم الصالحة في forecast/d1.json و forecast/d5_d10.json، والبوت بيقرا منهم.
خفيف عن قصد (مكتبات عادية بس) عشان يشتغل جوه monitor_once في Telegram Commands.
"""

import difflib
import json
import os
import re
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FC_DIR = os.path.join(ROOT, "forecast")
NAMES_FILE = os.path.join(ROOT, "news", "names.json")
AR_DAYS = {6: "الأحد", 0: "الاتنين", 1: "التلات", 2: "الأربع", 3: "الخميس", 4: "الجمعة", 5: "السبت"}

# أسماء عربي شائعة (بتتضاف لأي اسم عربي محفوظ في news/names.json)
ALIASES = {
    "COMI": ["التجاري الدولي", "البنك التجاري الدولي", "cib"], "FWRY": ["فوري"], "TMGH": ["طلعت مصطفى"],
    "HRHO": ["هيرميس", "اي اف جي هيرميس", "efg"], "ABUK": ["ابو قير", "ابوقير", "ابو قير للاسمدة"],
    "ETEL": ["المصرية للاتصالات", "تيليكوم مصر"], "SWDY": ["السويدي", "السويدي اليكتريك"],
    "ORAS": ["اوراسكوم للانشاء", "اوراسكوم كونستراكشون"], "ORHD": ["اوراسكوم للتنمية"],
    "EAST": ["الشرقية للدخان", "ايسترن كومباني", "ايسترن"], "MFPC": ["موبكو", "مصر لانتاج الاسمدة"],
    "EFIH": ["اي فاينانس"], "ADIB": ["مصرف ابوظبي الاسلامي"], "JUFO": ["جهينة"], "PHDC": ["بالم هيلز"],
    "SKPC": ["سيدي كرير"], "ISPH": ["ابن سينا فارما"], "AMOC": ["اموك"], "EGAL": ["مصر للالومنيوم"],
    "CLHO": ["كليوباترا هوسبيتال", "مستشفى كليوباترا"], "CCAP": ["القلعة", "سيتادل كابيتال"],
    "HELI": ["مصر الجديدة للاسكان"], "EMFD": ["اعمار مصر"], "ETRS": ["ايجيترانس"],
    "FAIT": ["بنك فيصل", "فيصل الاسلامي"], "CIEB": ["كريدي اجريكول"], "ALCN": ["الاسكندرية للحاويات"],
    "CPCI": ["القاهرة للادوية"], "AJWA": ["اجواء"], "OIH": ["اوراسكوم للاستثمار"],
}


def _norm(s):
    s = (s or "").lower()
    s = re.sub(r"[ً-ْـ]", "", s)               # تشكيل وتطويل
    s = re.sub(r"[أإآ]", "ا", s).replace("ة", "ه").replace("ى", "ي")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _names():
    try:
        with open(NAMES_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def find_ticker(text, bases):
    """text: رسالة المستخدم، bases: أكواد القايمة بدون .CA. بيرجّع (ticker, None) أو (None, [مقترحات])."""
    bset = set(bases)
    up = re.findall(r"[A-Za-z0-9]{2,8}", text.upper())
    hit = [w for w in up if w in bset]
    if hit:
        return hit[0], None
    q = _norm(text)
    if len(q) < 2:
        return None, []
    names = _names()
    cands = {}
    for t in bases:
        pool = [_norm(a) for a in ALIASES.get(t, [])]
        nm = names.get(t, {})
        pool += [_norm(nm.get("ar")), _norm(nm.get("en"))]
        for p in [x for x in pool if x]:
            if q == p:
                cands[t] = 3
            elif len(q) >= 3 and (q in p or (len(p) >= 4 and p in q)):
                cands[t] = max(cands.get(t, 0), 2)
            elif len(q) >= 4 and all(w in p.split() or w in p for w in q.split()):
                cands[t] = max(cands.get(t, 0), 1)
    if cands:
        best = max(cands.values())
        top = [t for t, v in cands.items() if v == best]
        if len(top) == 1:
            return top[0], None
        return None, sorted(top)[:6]
    close = difflib.get_close_matches(text.upper().strip(), list(bset), n=4, cutoff=0.6)
    return None, close


def _load(name):
    try:
        with open(os.path.join(FC_DIR, name), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _px(x):
    return f"{x:.3f}" if x < 1 else f"{x:.2f}"


def _lab(l):
    return {"شراء": "🟢 شراء", "بيع": "🔴 بيع", "انتظر": "⚪ انتظر", "احتفاظ": "🔵 احتفاظ"}.get(l, "❔")


SWING_TXT = {"دخول": "🟢 ادخل عند افتتاح الجلسة الجاية", "احتفاظ": "🔵 احتفاظ (مركز مفتوح في المحفظة التجريبية)",
             "خروج": "🔴 اخرج عند افتتاح الجلسة الجاية", "مراقبة": "⚪ مش مستوفي شروط الدخول دلوقتي"}


def swing_lines(base):
    """قرار محرك الدخول/الخروج (signals_report.py) للسهم ده."""
    sw = _load("swing.json")
    st = (sw or {}).get("stocks", {}).get(base)
    if not sw:
        return []
    if not st:
        return [f"🧭 قرار الدخول/الخروج (قفلة {sw.get('data_date')}): السهم مش في الموديل (سيولة قليلة أو بيانات ناقصة)."]
    top = max(1, round((1 - st["score"]) * 100))
    out = [f"🧭 قرار الدخول/الخروج (قفلة {sw.get('data_date')}): {SWING_TXT.get(st['status'], st['status'])}",
           f"   ترتيبه في قواعد البوت: أعلى {top}% من الأسهم"
           + ("" if sw.get("regime_on", True) else " | ⚠️ السوق في اتجاه هابط: مفيش دخول جديد")]
    return out


def forecast_text(base, news_info=None, now=None):
    """رسالة التوقعات لسهم واحد (بدون التحليل الفني اللي البوت بيعمله أصلاً)."""
    now = now or datetime.now()
    d1, d5 = _load("d1.json"), _load("d5_d10.json")
    nm = _names().get(base, {})
    title = f"🔮 توقعات {base}" + (f" - {nm['ar']}" if nm.get("ar") else (f" - {nm['en']}" if nm.get("en") else ""))
    L = [title]
    L += swing_lines(base)
    s1 = (d1 or {}).get("stocks", {}).get(base)
    if d1 and s1:
        tgt = d1.get("target")
        wd = AR_DAYS[datetime.strptime(tgt, "%Y-%m-%d").weekday()] if tgt else ""
        L.append(f"(بيانات قفلة {d1.get('data_date')} | آخر سعر {_px(s1['last'])})")
        L.append("")
        L.append(f"🗓 جلسة {wd} {tgt}: {_lab(s1['label'])} | ترتيب {s1['rank']} من {d1['n']}")
        if s1.get("lo") is not None:
            L.append(f"   نطاق الجلسة المعتاد: {_px(s1['lo'])} - {_px(s1['hi'])}")
        lv = s1.get("lv")
        if lv:
            L.append(f"   RSI {lv['rsi']:.0f} | تذبذب يومي {lv['atr_pct']:.1f}% | دعم {_px(lv['support'])} | مقاومة {_px(lv['resist'])}")
            L.append(f"   وقف خسارة {_px(lv['stop'])} | هدف {_px(lv['target'])} | عائد/مخاطرة {lv['rr']:.1f}")
    else:
        L.append("🗓 جلسة بكره: السهم مش في الموديل النهاردة (سيولة قليلة، أو قافل على الحد، أو بيانات Yahoo ناقصة).")
    accs = []
    if d1 and d1.get("hold"):
        accs.append(("بكره", d1["hold"]))
    for h in ("5", "10", "20", "40", "60"):
        blk = ((d5 or {}).get("h", {}) or {}).get(h)
        sx = (blk or {}).get("stocks", {}).get(base)
        if sx:
            L.append(f"🗓 {h} أيام: {_lab(sx['label'])} | ترتيب {sx['rank']} من {blk['n']}")
            if blk.get("hold"):
                accs.append((f"{h} أيام", blk["hold"]))
        else:
            L.append(f"🗓 {h} أيام: السهم مش في الموديل النهاردة.")
    if news_info is not None:
        try:
            import news as nw
            lab = (s1 or {}).get("label")
            L += nw.news_lines(news_info, lab)
        except Exception:  # noqa: BLE001
            pass
    if accs:
        L.append("")
        L.append("🎯 دقة شراء/بيع في آخر 250 يوم (على أيام الموديل ماشافهاش): " +
                 " | ".join(f"{k} {h['acc']:.0f}% (التخمين الثابت {h['base']:.0f}%)" for k, h in accs))
    stale = None
    dd = (d1 or {}).get("data_date") or (d5 or {}).get("data_date")
    if dd:
        age = (now.date() - datetime.strptime(dd, "%Y-%m-%d").date()).days
        if age >= 3:
            stale = f"⚠️ التوقعات دي من قفلة {dd} (قديمة {age} أيام)، والتقرير اليومي لسه ماحدّثهاش."
    if stale:
        L.append(stale)
    if not d1 and not d5:
        L.append("لسه مفيش توقعات متسجلة (التقرير اليومي ماشتغلش لحد دلوقتي).")
    L += ["", "⚠️ قرار 🧭 هو المعتمد للدخول والخروج (اتختبر كمحفظة كاملة). ترتيبات الجلسة و5/10 أيام إحصائية ضعيفة "
              "ومش توصية. القرار والمسؤولية ليك."]
    return "\n".join(L)


# ============================================================
# رد "كود سهم" الجديد: رد واحد مبني على قواعد البوت الحية وأرقام من التاريخ الفعلي بس
# ============================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _acc():
    try:
        with open(os.path.join(ROOT_DIR, "signals", "account.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _desc(base):
    try:
        with open(os.path.join(ROOT_DIR, "data", "sectors.json"), encoding="utf-8") as fh:
            d = json.load(fh).get(base, {})
        return d.get("desc"), d.get("sector")
    except (OSError, ValueError):
        return None, None


def stock_view(base, live=None, news_info=None):
    """live: {close, prev, date} من TradingView (اختياري)."""
    sw = _load("swing.json")
    desc, sector = _desc(base)
    head = f"🔎 {base}" + (f" - {desc}" if desc else "") + (f" ({sector})" if sector and sector != "?" else "")
    if not sw:
        return head + "\nبيانات البوت لسه متعملتش (بتتحدث كل يوم بعد القفل)."
    st = sw.get("stocks", {}).get(base)
    p = sw.get("params", {})
    L = [head]
    if not st or st.get("close") is None:
        return "\n".join(L + ["مفيش بيانات أسعار كفاية للسهم ده (ممكن يكون موقوف أو جديد)."])
    px = f"قفلة {sw['data_date']}: {_px(st['close'])}"
    if live and live.get("close"):
        ch = (live["close"] / live["prev"] - 1) * 100 if live.get("prev") else None
        px += f" | دلوقتي {_px(live['close'])}" + (f" ({ch:+.1f}%)" if ch is not None else "")
    L.append(px)
    L.append("")
    acc = _acc() or {}
    pos = acc.get("pos", {}).get(base)
    status = st["status"]
    q_in = p.get("q_in", 0.9)
    top = max(1, round((1 - st["score"]) * 100)) if st.get("score") is not None else None
    need_top = round((1 - q_in) * 100)
    # 1) القرار
    if status == "دخول":
        L.append("🧭 قرار البوت: 🟢 هيشتريه عند افتتاح الجلسة الجاية")
    elif status == "خروج":
        L.append("🧭 قرار البوت: 🔴 هيبيعه عند افتتاح الجلسة الجاية")
    elif pos:
        L.append("🧭 قرار البوت: 🔵 معانا في المحفظة - احتفاظ")
    else:
        L.append("🧭 قرار البوت: ⚪ مش هيشتريه دلوقتي")
    # 2) الشروط
    if pos:
        cur = live["close"] if live and live.get("close") else pos["last_px"]
        pnl = pos["shares"] * cur - pos["cost"]
        k = p.get("trail_k") or 0
        stop = pos["peak_px"] * (1 - k * pos["atr"]) if k else None
        L += [f"• معانا {pos['shares']:,} سهم من {pos['entry_date']} بسعر {_px(pos['entry_px'])} → "
              f"{'+' if pnl >= 0 else '-'}{abs(pnl):,.0f} جنيه ({pnl / pos['cost'] * 100:+.1f}%)"]
        if stop:
            L.append(f"• الوقف المتحرك {_px(stop)} (بعيد {(cur / stop - 1) * 100:.1f}%): لو قفل تحته يتباع الافتتاح اللي بعده")
        L.append(f"• بيتباع كمان لو ترتيبه نزل تحت أعلى {round((1 - p.get('q_out', 0.3)) * 100)}% "
                 f"(دلوقتي أعلى {top}%) أو عدّى {p.get('max_hold', 60)} جلسة (عدّى {pos['days']})")
    else:
        ok = lambda b: "✅" if b else "❌"  # noqa: E731
        L.append("✅ سيولته كفاية للتداول" if st["liquid"] else "❌ سيولته قليلة (أقل من مليون جنيه تداول في اليوم) أو مقفول على الحد")
        if top is not None:
            L.append(f"{ok(st['score'] >= q_in)} ترتيبه: أعلى {top}% من الأسهم (الشرط: من أعلى {need_top}%)")
        if p.get("trend50"):
            L.append("✅ السهم فوق متوسط 50 جلسة (اتجاهه صاعد)" if st["sma50"] else
                     "❌ السهم تحت متوسط 50 جلسة (لازم يبقى فوقه)")
        if p.get("regime"):
            need = sw.get("regime_need")
            L.append("✅ السوق في اتجاه صاعد" if sw.get("regime_on") else
                     "❌ السوق في اتجاه هابط" + (f" (مؤشر السوق محتاج يطلع {need:+.1f}% عشان البوت يشتري أي سهم)" if need else ""))
        allok = st["liquid"] and st.get("score", 0) >= q_in and (st["sma50"] or not p.get("trend50")) and sw.get("regime_on")
        if allok and status == "مراقبة":
            L.append("ℹ️ مستوفي الشروط، بس المحفظة مليانة أو فيه أسهم ترتيبها أعلى منه.")
    # 3) من التاريخ: الأسهم اللي كانت في نفس الترتيب
    hist = sw.get("hist") or {}
    reg = "on" if sw.get("regime_on") else "off"
    if st.get("score") is not None and hist.get(reg):
        d = str(min(9, int(st["score"] * 10)))
        h = hist[reg].get(d)
        if h:
            L += ["", f"📊 من التاريخ (2012-{sw['data_date'][:4]}): الأسهم اللي كانت في نفس ترتيبه "
                      f"والسوق {'صاعد' if reg == 'on' else 'هابط'} زي دلوقتي ({h['n']:,} حالة)، بعد {sw.get('hist_h', 40)} جلسة:",
                  f"   طلعت في {h['win']:.0f}% من المرات | المتوسط {h['avg']:+.1f}% والوسيط {h['med']:+.1f}% | "
                  f"الفرق عن السوق {h['exc']:+.1f}%"]
            best = hist[reg].get("9")
            if best and d != "9":
                L.append(f"   (للمقارنة: أعلى 10% في الترتيب: طلعت {best['win']:.0f}% من المرات، المتوسط {best['avg']:+.1f}%)")
    tr = st.get("track")
    if tr:
        L.append(f"🤖 البوت بنفس القواعد عمل على السهم ده {tr['n']} صفقة من 2012: كسب في {tr['win']}، "
                 f"متوسط الصفقة {tr['avg']:+.1f}%" + (f" | آخر مرة {tr['last'][0]} → {tr['last'][1]}: {tr['last'][2]:+.1f}%"
                                                     if tr.get("last") else ""))
    elif st.get("score") is not None:
        L.append("🤖 البوت بنفس القواعد ماشتراش السهم ده قبل كده في الباكتست (2012+).")
    # 4) أرقام السهم
    L += ["", f"📈 تذبذبه اليومي {st['atr']:.1f}% | تكلفة الشراء+البيع حوالي {st['spread'] * 2 + 0.3:.1f}% "
              + (f"| بعيد عن أعلى سعر في سنة {st['from_hi']:+.0f}%" if st.get("from_hi") is not None else "")]
    if news_info and news_info.get("head"):
        t_, src, age = news_info["head"]
        L.append(f"📰 آخر خبر: {t_} ({src}{'، ' + str(age) if age else ''})")
    L.append("⚠️ أرقام من تاريخ السوق الفعلي، مش ضمان للجاي.")
    return "\n".join(L)
