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
    return {"شراء": "🟢 شراء", "بيع": "🔴 بيع", "انتظر": "⚪ انتظر"}.get(l, "❔")


def forecast_text(base, news_info=None, now=None):
    """رسالة التوقعات لسهم واحد (بدون التحليل الفني اللي البوت بيعمله أصلاً)."""
    now = now or datetime.now()
    d1, d5 = _load("d1.json"), _load("d5_d10.json")
    nm = _names().get(base, {})
    title = f"🔮 توقعات {base}" + (f" - {nm['ar']}" if nm.get("ar") else (f" - {nm['en']}" if nm.get("en") else ""))
    L = [title]
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
    for h in ("5", "10"):
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
    L += ["", "⚠️ شراء/بيع = تقدير إحصائي ضعيف (دقته قريبة من التخمين الثابت)، الترتيب نسبي بين الأسهم، "
              "ومش توصية. القرار والمسؤولية ليك.",
          "(الترتيب والتصنيف حاجتين منفصلتين: ممكن سهم ترتيبه ضعيف بين الأسهم وتصنيفه شراء لو الموديل شايف الكل طالع)"]
    return "\n".join(L)
