"""
أخبار الأسهم المرشحة: بيجيب آخر عناوين كل سهم (Google News، عربي وإنجليزي) وبيطلّع لكل سهم:
  - أثر الأخبار: إيجابي / سلبي / محايد / مفيش (Gemini بيقرأ العناوين بس، من غير معلومات من برا)
  - أهم عنوان + مصدره وعمره
  - هل الأخبار بتأيد تصنيف شراء/بيع ولا بتعارضه

⚠️ الأخبار هنا بتأكيد/تحذير بس: مبتغيّرش التصنيف ولا الترتيب، لأن مفيش باكتست للأخبار.
بنسجّل لكل ترشيح هل الأخبار أيدت التصنيف ولا عارضته، وبعد كام جلسة رسالة النتيجة بتقولك لو الأخبار فعلاً
بتفرق (لما أيدت: التصنيف صح كام مرة، ولما عارضت: صح كام مرة).

لو Gemini مش متاح (مفتاح ناقص أو فشل): بنعرض العناوين من غير تقييم، ومفيش تأييد/معارضة.
لو الأخبار كلها فشلت: التقرير بيكمّل من غيرها.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from xml.etree import ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NEWS_DIR = os.path.join(ROOT, "news")
NAMES_FILE = os.path.join(NEWS_DIR, "names.json")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
WINDOW_DAYS = 7
MAX_HEADLINES = 6
UA = {"User-Agent": "Mozilla/5.0 (compatible; egx-news-bot)"}
ICON = {"إيجابي": "🟢", "سلبي": "🔴", "محايد": "⚪", "مفيش": "▫️"}
BUY, SELL = "شراء", "بيع"


# ============================================================
# أسماء الشركات (كاش في news/names.json)
# ============================================================
def _load_names():
    try:
        with open(NAMES_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_names(d):
    os.makedirs(NEWS_DIR, exist_ok=True)
    with open(NAMES_FILE, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, sort_keys=True)


def _gemini(prompt):
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None
    from google import genai
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return (resp.text or "").strip()


def _json_from(text):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except ValueError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                return None
    return None


def resolve_names(tickers):
    """بيرجّع {ticker: {"en":..., "ar":...}}. الاسم الإنجليزي من Yahoo، والعربي من Gemini (للبحث بس)."""
    names = _load_names()
    need = [t for t in tickers if t not in names or not names[t].get("en")]
    if need:
        import yfinance as yf
        for t in need:
            en = None
            try:
                info = yf.Ticker(t + ".CA").info or {}
                en = info.get("longName") or info.get("shortName")
            except Exception as e:  # noqa: BLE001
                print(f"[news] اسم {t}: {str(e)[:80]}")
            names[t] = {"en": en, "ar": names.get(t, {}).get("ar")}
            time.sleep(0.3)
    need_ar = [t for t in tickers if names.get(t, {}).get("en") and not names[t].get("ar")]
    if need_ar:
        try:
            lst = "\n".join(f"{t}: {names[t]['en']}" for t in need_ar)
            out = _json_from(_gemini(
                "دي شركات مقيدة في البورصة المصرية (الكود: الاسم بالإنجليزي). اكتب لكل شركة اسمها العربي الأشهر "
                "اللي بيتكتب بيه في الأخبار المصرية، من غير كلمة 'شركة' ولا 'ش.م.م'. لو مش متأكد اكتب null. "
                "رجّع JSON بس بالشكل {\"الكود\": \"الاسم\"}.\n" + lst))
            if isinstance(out, dict):
                for t in need_ar:
                    v = out.get(t)
                    if isinstance(v, str) and v.strip():
                        names[t]["ar"] = v.strip()
        except Exception as e:  # noqa: BLE001
            print(f"[news] أسماء عربي فشلت: {str(e)[:120]}")
    if need or need_ar:
        _save_names(names)
    return names


# ============================================================
# جلب العناوين
# ============================================================
def parse_rss(xml_text, now, window_days=WINDOW_DAYS):
    """بيرجّع [(title, source, age_days)] لعناوين آخر window_days يوم."""
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        src = (it.findtext("source") or "").strip()
        if not src and " - " in title:
            title, src = title.rsplit(" - ", 1)
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate") or "")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        age = (now - dt).total_seconds() / 86400
        if -1 <= age <= window_days:
            out.append((title.strip(), src.strip(), max(age, 0.0)))
    return out


def _rss(query, lang, now):
    import requests
    if lang == "ar":
        loc = "hl=ar&gl=EG&ceid=EG:ar"
    else:
        loc = "hl=en-US&gl=EG&ceid=EG:en"
    url = f"https://news.google.com/rss/search?q={quote(query)}&{loc}"
    r = requests.get(url, headers=UA, timeout=15)
    r.raise_for_status()
    return parse_rss(r.text, now)


def fetch_headlines(ticker, nm, now):
    """عناوين السهم: بحث عربي بالاسم العربي + بحث إنجليزي بالاسم الإنجليزي (مدموجين بدون تكرار)."""
    qs = []
    if nm.get("ar"):
        qs.append((f"\"{nm['ar']}\" سهم OR البورصة when:{WINDOW_DAYS}d", "ar"))
    if nm.get("en"):
        en = re.sub(r"\b(S\.?A\.?E\.?|Co\.?|Company|Ltd\.?|\(Egypt\)|Egypt)\b", "", nm["en"], flags=re.I).strip(" ,.-")
        qs.append((f"\"{en}\" Egypt when:{WINDOW_DAYS}d", "en"))
    if not qs:
        qs.append((f"{ticker} البورصة المصرية when:{WINDOW_DAYS}d", "ar"))
    seen, items = set(), []
    for q, lang in qs:
        try:
            for title, src, age in _rss(q, lang, now):
                k = re.sub(r"\W+", "", title.lower())[:60]
                if k not in seen:
                    seen.add(k)
                    items.append((title, src, age))
        except Exception as e:  # noqa: BLE001
            print(f"[news] {ticker} ({lang}): {str(e)[:100]}")
        time.sleep(0.4)
    items.sort(key=lambda x: x[2])
    return items[:MAX_HEADLINES]


# ============================================================
# تقييم الأثر (Gemini)
# ============================================================
def assess(batch, names):
    """batch: {ticker: [(title, source, age)]}. بيرجّع {ticker: (sentiment, why, idx_of_key_headline)}."""
    have = {t: h for t, h in batch.items() if h}
    if not have:
        return {}
    blocks = []
    for t, h in have.items():
        nm = names.get(t, {})
        blocks.append(f"### {t} ({nm.get('ar') or ''} / {nm.get('en') or ''})\n" +
                      "\n".join(f"{i}. {title} [{src}، من {age:.0f} يوم]" for i, (title, src, age) in enumerate(h)))
    prompt = (
        "انت محلل أخبار أسهم في البورصة المصرية. لكل سهم تحت عناوين أخبار آخر أسبوع. اعتمد على العناوين دي "
        "بس، من غير أي معلومة من برا ومن غير تخمين.\n"
        "حدد أثر الأخبار على سعر السهم الأيام الجاية: \"إيجابي\" (نتايج أحسن/توزيعات/صفقة/عقد/توسع/ترقية)، "
        "\"سلبي\" (خسائر/غرامة/قضية/تخفيض/أزمة/نقص مواد)، \"محايد\" (أخبار موجودة بس مش مؤثرة أو متضاربة)، "
        "أو \"مفيش\" لو مفيش عنوان عن الشركة نفسها (أخبار السوق العامة أو شركات تانية تتجاهلها).\n"
        "رجّع JSON بس بالشكل: {\"الكود\": {\"s\": \"إيجابي\", \"why\": \"جملة قصيرة بالمصري\", \"i\": رقم أهم عنوان}}\n\n"
        + "\n\n".join(blocks))
    out = _json_from(_gemini(prompt))
    res = {}
    if isinstance(out, dict):
        for t in have:
            v = out.get(t)
            if isinstance(v, dict) and v.get("s") in ICON:
                try:
                    i = int(v.get("i", 0))
                except (TypeError, ValueError):
                    i = 0
                res[t] = (v["s"], str(v.get("why", "")).strip()[:140], i if 0 <= i < len(have[t]) else 0)
    return res


def news_for(tickers, now=None):
    """الدالة الرئيسية: بترجّع {ticker: {"n", "sent", "why", "head": (title, src, age)}} أو {} لو فشل كل حاجة."""
    now = now or datetime.now(timezone.utc)
    tickers = list(dict.fromkeys(tickers))
    try:
        names = resolve_names(tickers)
        batch = {t: fetch_headlines(t, names.get(t, {}), now) for t in tickers}
    except Exception as e:  # noqa: BLE001
        print(f"[news] فشل الجلب: {e}")
        return {}
    try:
        verdict = assess(batch, names)
    except Exception as e:  # noqa: BLE001
        print(f"[news] تقييم Gemini فشل، هعرض العناوين من غير تقييم: {str(e)[:150]}")
        verdict = {}
    out = {}
    for t in tickers:
        h = batch.get(t, [])
        v = verdict.get(t)
        info = {"n": len(h), "sent": v[0] if v else None, "why": v[1] if v else "",
                "head": h[v[2]] if (v and h) else (h[0] if h else None)}
        if info["sent"] == "مفيش":
            info["head"] = None
        out[t] = info
    return out


# ============================================================
# العرض والتأييد/المعارضة
# ============================================================
def agreement(label, sent):
    """'agree' لو الأخبار في نفس اتجاه التصنيف، 'disagree' لو عكسه، 'neutral' غير كده، None لو مفيش تقييم."""
    if not label or sent is None:
        return None
    if sent in ("محايد", "مفيش"):
        return "neutral"
    if (label == BUY and sent == "إيجابي") or (label == SELL and sent == "سلبي"):
        return "agree"
    return "disagree"


def _age(a):
    return "النهاردة" if a < 1 else ("من يوم" if a < 2 else f"من {a:.0f} أيام")


def news_lines(info, label):
    """سطور الرسالة لسهم واحد (مسافة بادئة 3). info ممكن تبقى None لو الأخبار كلها فشلت."""
    if info is None:
        return []
    if info["n"] == 0 or info["sent"] == "مفيش":
        return [f"   📰 مفيش أخبار عن السهم في آخر {WINDOW_DAYS} أيام"]
    head = info["head"]
    htxt = f"«{head[0][:90]}» ({head[1] or 'مصدر؟'}، {_age(head[2])})" if head else ""
    if info["sent"] is None:
        return [f"   📰 {htxt} (من غير تقييم)"]
    ag = agreement(label, info["sent"])
    tag = {"agree": " ✅ بتأيد التصنيف", "disagree": " ⚠️ عكس التصنيف", "neutral": ""}.get(ag, "")
    l1 = f"   📰 {ICON[info['sent']]} {info['sent']}{' - ' + info['why'] if info['why'] else ''}{tag}"
    return [l1] + ([f"      {htxt}"] if htxt else [])


def agreement_summary(days):
    """سطر الحصيلة: التصنيف صح كام مرة لما الأخبار أيدت / عارضت / كانت محايدة."""
    st = {"agree": [0, 0], "disagree": [0, 0], "neutral": [0, 0]}
    for d in days:
        for x in d.get("picks", []):
            nw, lab = x.get("nw"), x.get("lab")
            if nw in st and lab and x["r"] != 0:
                st[nw][1] += 1
                st[nw][0] += int((x["r"] > 0) == (lab == BUY))
    if not any(v[1] for v in st.values()):
        return None
    f = lambda k, nm: f"{nm}: صح {st[k][0]} من {st[k][1]}" if st[k][1] else f"{nm}: لسه مفيش"  # noqa: E731
    return "📰 التصنيف لما الأخبار " + " | ".join([f("agree", "أيدت"), f("disagree", "عارضت"), f("neutral", "محايدة/مفيش")])
