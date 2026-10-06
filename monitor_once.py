"""
================================================================
   EGX Alert Monitor - أداة مراقبة وتنبيه بس (مش تنفيذ)
================================================================
البوت ده **مش بينفذ ولا صفقة**، حقيقية ولا حتى وهمية. دوره الوحيد:
يراقب الأسهم في القايمة، ولما إشارة الترند المُثبتة (نفس منطق
trend_following اللي اتأكدنا منه بالباكتست وخارج العينة) تتغيّر
فعليًا، يبعتلك تنبيه على تليجرام - وانت تقرر تشتري/تبيع بنفسك من
حسابك الحقيقي أو لأ.

منطق الإشارة (زي الباكتست بالظبط):
  - تنبيه شرا: السعر بقى فوق المتوسطين (20 و50 يوم) بعد ما ماكانش
  - تنبيه بيع/تحذير: السعر كسر المتوسط الطويل (50 يوم) بعد ما كان
    البوت بعتلك تنبيه شرا قبل كده على نفس السهم

مفيش كاش، مفيش "درجة ثقة"، مفيش محفظة وهمية - الملف ده أبسط بكتير
من نسخة التداول الوهمي اللي كانت شغالة قبل كده، لأن مفيش داعي لكل
حسابات المحفظة لما مفيش تنفيذ أصلاً.

⚠️ التنبيهات دي مبنية على استراتيجية اتأكدنا إنها بتحقق ميزة صغيرة
بس حقيقية (مش overfitting) - لكنها برضو بتاخد جزء صغير من أي صعود
قوي، ومفيش ضمان ربح. راجع كل تنبيه بنفسك قبل أي قرار حقيقي.
================================================================
"""

import os
import json
import csv
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import requests

# مكونات EGX 30 (31 خط تداول - فالمور ليها خطين VLMR و VLMRA). القايمة بتتغير
# مع إعادة تشكيل المؤشر، فالمصدر الأساسي هو TICKERS في monitor.yml
EGX30_DEFAULT = ("ABUK.CA,ADIB.CA,ALCN.CA,AMOC.CA,BTFH.CA,CCAP.CA,CLHO.CA,COMI.CA,"
                 "EAST.CA,EFID.CA,EFIH.CA,EGAL.CA,EMFD.CA,ETEL.CA,FWRY.CA,GBCO.CA,"
                 "HELI.CA,HRHO.CA,ISPH.CA,JUFO.CA,MCQE.CA,MFPC.CA,ORAS.CA,ORHD.CA,"
                 "PHDC.CA,RAYA.CA,RMDA.CA,SKPC.CA,TMGH.CA,VLMR.CA,VLMRA.CA")

TICKERS_FILE = "tickers.txt"


def load_tickers():
    """
    الأولوية: متغير TICKERS (لو متحدد) ثم ملف tickers.txt ثم قايمة EGX 30.
    ملف tickers.txt بيتقري بـ regex: أي كود بالشكل XXXX.CA بيتاخد، فتقدر
    تلزق فيه الجدول كله زي ما هو من موقع EGX (ISIN وأسماء وأوزان) والتكرار
    بيتشال لوحده. أي سطر بيبدأ بـ # بيتتجاهل.
    """
    import re
    raw = os.environ.get("TICKERS", "").strip()
    if not raw and os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "r", encoding="utf-8") as f:
            raw = "\n".join(l for l in f if not l.lstrip().startswith("#"))
    if not raw:
        raw = EGX30_DEFAULT
    seen, out = set(), []
    for t in re.findall(r"[A-Za-z0-9]{2,8}\.CA", raw):
        t = t.upper()
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# GGRN اتشال: مش مغطى في yfinance/Yahoo Finance خالص
TICKERS = load_tickers()

LOOKBACK_WINDOW = 20
TREND_FILTER_WINDOW = 50
# بنحتفظ بقفلات يومية (مش قراءات كل 15 دقيقة) - 60 يوم كفاية للـ SMA50
HISTORY_WINDOW = 60
STATE_FILE = "state.json"
ALERT_LOG_FILE = "alert_log.csv"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FORCE_RUN = os.environ.get("FORCE_RUN", "false").lower() == "true"

# لو السعر الجديد مختلف عن آخر سعر معروف بنسبة أكبر من الحد ده،
# هنتأكد منه عن طريق yfinance قبل ما نصدقه
PRICE_SANITY_THRESHOLD = 0.20

# لو الإشارات في run واحد أكتر من العدد ده، بتتبعت في رسالة واحدة مختصرة
ALERT_DIGEST_THRESHOLD = 5

# التقرير اليومي: بيتبعت لوحده مرة في اليوم بعد الساعة 2 الضهر بتوقيت القاهرة.
# DAILY_FOCUS = أسهم بتاخد تحليل كامل في التقرير (غيرها من env في monitor.yml)
DAILY_FOCUS = [t.strip().upper() for t in os.environ.get(
    "DAILY_FOCUS", "COMI.CA,EFIH.CA,EFID.CA,ABUK.CA").split(",") if t.strip()]
DAILY_REPORT_HOUR = 14      # بعد الساعة دي (القاهرة) في أيام التداول
NEAR_SIGNAL_PCT = 2.0       # "قريب من إشارة" = أقل من 2% بعيد عن المتوسط
TOP_N = 5


# ============================================================
# مواعيد تداول EGX
# ============================================================
def is_market_open_now():
    cairo = datetime.now(ZoneInfo("Africa/Cairo"))
    trading_days = {6, 0, 1, 2, 3}
    if cairo.weekday() not in trading_days:
        return False
    market_open = cairo.replace(hour=10, minute=0, second=0, microsecond=0)
    market_close = cairo.replace(hour=14, minute=30, second=0, microsecond=0)
    return market_open <= cairo <= market_close


# ============================================================
# الحالة المحفوظة - بس قفلات الأيام وآخر إشارة اتبعتت، من غير أي
# محفظة أو كاش
# ============================================================
def default_stock_state():
    # daily = قفلة كل يوم تداول، dates = تاريخ كل قفلة (بتوقيت القاهرة)
    # initialized = السهم اتعمله تهيئة صامتة (من غير تنبيه) أول ما التاريخ اكتمل
    return {"dates": [], "daily": [], "last_signal": "NONE",  # NONE / BUY / SELL
            "initialized": False}


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {}
    stocks = raw.get("stocks", {})
    for ticker in TICKERS:
        if ticker not in stocks:
            stocks[ticker] = default_stock_state()
        else:
            stocks[ticker].setdefault("last_signal", "NONE")
            stocks[ticker].setdefault("dates", [])
            stocks[ticker].setdefault("daily", [])
            stocks[ticker].pop("prices", None)  # تنسيق قديم (قراءات كل 15 دقيقة)
            # الأسهم اللي عندها تاريخ كامل بالفعل متحسبش جديدة
            stocks[ticker].setdefault(
                "initialized", len(stocks[ticker]["daily"]) >= TREND_FILTER_WINDOW)
    return {"stocks": stocks, "meta": raw.get("meta", {})}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def append_alert_log(ticker, signal, price, sma20, sma50):
    file_exists = os.path.exists(ALERT_LOG_FILE)
    with open(ALERT_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "ticker", "signal", "price", "sma20", "sma50"])
        writer.writerow([
            datetime.now(ZoneInfo("Africa/Cairo")).strftime("%Y-%m-%d %H:%M:%S"),
            ticker, signal, f"{price:.2f}", f"{sma20:.2f}", f"{sma50:.2f}",
        ])


# ============================================================
# جلب السعر (نفس منطق الفحص القديم - Mubasher أول، yfinance احتياطي،
# مع فحص أمان للقفزات الغريبة)
# ============================================================
def fetch_price_mubasher(ticker):
    import re
    symbol = ticker.replace(".CA", "")
    url = f"https://english.mubasher.info/markets/EGX/stocks/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    match = re.search(
        r'market time\.(?:(?!\d{1,3}\.\d{1,2}).){0,300}?(\d{1,3}\.\d{1,2})',
        resp.text, re.DOTALL,
    )
    if not match:
        raise ValueError("مش قادر ألاقي السعر في صفحة Mubasher")
    return float(match.group(1))


def fetch_price_yfinance(ticker):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    try:
        price = tk.fast_info["last_price"]
        if price is not None and price == price:
            return float(price)
    except Exception:
        pass
    data = tk.history(period="5d", interval="1d")
    if data.empty:
        raise ValueError("مفيش بيانات متاحة")
    return float(data.iloc[-1]["Close"])


def fetch_price(ticker, prev_price=None):
    try:
        price = fetch_price_mubasher(ticker)
    except Exception as e:
        print(f"  [Mubasher] فشل الجلب ({e})، هجرب yfinance كـ fallback")
        return fetch_price_yfinance(ticker)

    if prev_price and prev_price > 0:
        change = abs(price - prev_price) / prev_price
        if change > PRICE_SANITY_THRESHOLD:
            print(f"  [تنبيه] قفزة سعر غريبة من Mubasher ({change:.1%})، بيتأكد من yfinance...")
            try:
                yf_price = fetch_price_yfinance(ticker)
                yf_change = abs(yf_price - prev_price) / prev_price
                if yf_change < change * 0.5:
                    print(f"  [تنبيه] سعر Mubasher يبدو غير موثوق، هستخدم yfinance بدالاً: {yf_price:.2f}")
                    return yf_price
            except Exception as e2:
                print(f"  [تنبيه] فشل التأكد من yfinance ({e2})، هكمل بسعر Mubasher زي ما هو")

    return price


# ============================================================
# التنبيه والشرح
# ============================================================
def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[تليجرام] التوكن أو chat_id فاضيين")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
        if resp.status_code != 200:
            print("[تليجرام] فشل الإرسال:", resp.text)
    except Exception as e:
        print("[تليجرام] خطأ:", e)


def fallback_explanation(signal, price, sma20, sma50):
    if signal == "BUY":
        return (f"السعر ({price:.2f}) بقى فوق المتوسط القريب ({sma20:.2f}) "
                f"والمتوسط الطويل ({sma50:.2f}) مع بعض - إشارة ترند صاعد جديد.")
    return (f"السعر ({price:.2f}) كسر المتوسط الطويل ({sma50:.2f}) - "
            f"إشارة إن الترند الصاعد ضعف أو انكسر.")


def explain_alert(signal, price, sma20, sma50):
    if not GEMINI_API_KEY:
        return fallback_explanation(signal, price, sma20, sma50)
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""اكتب جملتين بس بالعربي البسيط (مصري) تشرح لمستثمر عادي
ليه ظهرت الإشارة دي، من غير مقدمة ولا خاتمة:

الإشارة: {"بداية ترند صاعد" if signal == "BUY" else "انكسار الترند الصاعد"}
السعر الحالي: {price:.2f}
المتوسط القريب (20 يوم): {sma20:.2f}
المتوسط الطويل (50 يوم): {sma50:.2f}
"""
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (resp.text or "").strip()
        return text if text else fallback_explanation(signal, price, sma20, sma50)
    except Exception as e:
        print("  [Gemini] فشل، هستخدم شرح تلقائي:", e)
        return fallback_explanation(signal, price, sma20, sma50)


# ============================================================
# القفلات اليومية
# ============================================================
def update_daily(stock_state, price):
    today = datetime.now(ZoneInfo("Africa/Cairo")).strftime("%Y-%m-%d")
    dates = stock_state.setdefault("dates", [])
    daily = stock_state.setdefault("daily", [])
    if dates and dates[-1] == today:
        daily[-1] = price  # نفس اليوم: حدّث القفلة
    else:
        dates.append(today)
        daily.append(price)
    stock_state["dates"] = dates[-HISTORY_WINDOW:]
    stock_state["daily"] = daily[-HISTORY_WINDOW:]


def adjust_for_corporate_actions(values):
    """
    EGX عنده حد يومي ±10%، فأي قفزة أكبر من 20% في يوم واحد يبقى غالبًا
    تقسيم أو أسهم مجانية (مش حركة سعر حقيقية). بنعدّل الأسعار اللي قبل
    القفزة بنفس النسبة عشان المتوسطات ما تتبوظش.
    """
    a = [float(x) for x in values]
    for i in range(len(a) - 1, 0, -1):
        ratio = a[i] / a[i - 1]
        if ratio < 0.8 or ratio > 1.25:
            print(f"  [تعديل] قفزة بنسبة {ratio:.3f} عند القفلة رقم {i} - اتعدّلت الأسعار اللي قبلها")
            a[:i] = [x * ratio for x in a[:i]]
    return a


def seed_history(ticker, stock_state):
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="6mo", interval="1d")["Close"].dropna()
    if h.empty:
        raise ValueError("مفيش تاريخ متاح من yfinance")
    h = h.tail(HISTORY_WINDOW)
    stock_state["daily"] = adjust_for_corporate_actions(h)
    stock_state["dates"] = [d.strftime("%Y-%m-%d") for d in h.index]
    print(f"  [seed] اتملّى {len(stock_state['daily'])} يوم تاريخ من yfinance")


# ============================================================
# توقع الجلسة الجاية (تقدير إحصائي من توزيع التغيرات اليومية
# الفعلية للسهم في آخر ~60 يوم - مش نموذج تنبؤ ولا ضمان)
# ============================================================
def compute_outlook(daily, price, trend_up, trend_broken):
    arr = np.array(daily[-HISTORY_WINDOW:], dtype=float)
    rets = np.diff(arr) / arr[:-1]
    n = len(rets)
    if n < 20 or len(daily) < TREND_FILTER_WINDOW:
        return None

    # 1) احتمال صعود/هبوط: نسبة أيام الصعود والهبوط في تاريخ السهم نفسه،
    #    مع تهدئة ناحية 50% (K) عشان العينة صغيرة وماتديش ثقة زيادة.
    #    الأيام الثابتة (إجازات/سهم واقف) بتتشال من الحساب.
    K = 20
    ups = int((rets > 0).sum())
    downs = int((rets < 0).sum())
    p_up = (ups + K / 2) / (ups + downs + K) * 100
    p_down = 100 - p_up

    # 2) تأكيد الصعود فنيًا: كام مؤشر من 5 بيأيد الصعود
    sma20 = float(np.mean(daily[-LOOKBACK_WINDOW:]))
    sma50 = float(np.mean(daily[-TREND_FILTER_WINDOW:]))
    checks = [
        price > sma20,
        price > sma50,
        sma20 > sma50,
        price > daily[-6],
        price > daily[-21],
    ]
    k = sum(checks)
    tech = k / len(checks) * 100
    if tech >= 60:
        label = "صعود 📈"
    elif tech <= 40:
        label = "هبوط 📉"
    else:
        label = "محايد ➖"

    # 3) رينج السعر لو صعد / لو هبط: من توزيع حركات الصعود (أو الهبوط)
    #    الفعلية للسهم - من المئين 10 للمئين 90
    up_r = rets[rets > 0]
    dn_r = rets[rets < 0]
    if len(up_r) < 5:
        up_r = np.array([0.0, abs(rets).max()])
    if len(dn_r) < 5:
        dn_r = np.array([-abs(rets).max(), 0.0])
    up_lo, up_hi = np.percentile(up_r, [10, 90])
    dn_lo, dn_hi = np.percentile(dn_r, [10, 90])  # dn_lo = أكبر هبوط
    return {
        "label": label,
        "p_up": float(p_up),
        "p_down": float(p_down),
        "tech": float(tech),
        "tech_k": k,
        "up_low": price * (1 + up_lo),
        "up_high": price * (1 + up_hi),
        "down_low": price * (1 + dn_lo),
        "down_high": price * (1 + dn_hi),
        "days": n,
    }


# ============================================================
# معالجة سهم واحد - بس بيحسب الإشارة وبيقارنها بآخر إشارة معروفة،
# مفيش تنفيذ ولا محفظة خالص
# ============================================================
def process_stock(ticker, stock_state):
    if len(stock_state.get("daily", [])) < TREND_FILTER_WINDOW:
        try:
            seed_history(ticker, stock_state)
        except Exception as e:
            print(f"  فشل seed: {e}")

    daily = stock_state.get("daily", [])
    prev_price = daily[-1] if daily else None

    try:
        price = fetch_price(ticker, prev_price=prev_price)
    except Exception as e:
        print(f"  خطأ في جلب السعر: {e}")
        return None

    # سعر صفر أو سالب = بيانات غير صالحة (غالبًا سهم موقوف أو الصفحة مفيهاش سعر)
    if price is None or price <= 0:
        print(f"  [تجاهل] سعر غير صالح ({price}) - غالبًا السهم موقوف، هحاول في الـ run الجاي")
        return None

    # رفض القفزة الغريبة: متتخزنش ويتبعتلك تحذير (مرة واحدة في اليوم لكل سهم)
    if prev_price and abs(price - prev_price) / prev_price > PRICE_SANITY_THRESHOLD:
        print(f"  [رفض] قفزة غريبة {prev_price:.2f} -> {price:.2f}، اتجاهلت")
        today = datetime.now(ZoneInfo("Africa/Cairo")).strftime("%Y-%m-%d")
        if stock_state.get("spike_warned_on") != today:
            stock_state["spike_warned_on"] = today
            send_telegram(f"⚠️ {ticker.replace('.CA', '')}: قفزة سعر غريبة "
                          f"{prev_price:.2f} → {price:.2f}، اتجاهلت ومتخزنتش. راجعها يدويًا. "
                          f"لو ده تقسيم أو أسهم مجانية امسح السهم ده من state.json "
                          f"عشان يتعمله seed من جديد.")
        return None

    update_daily(stock_state, price)
    daily = stock_state["daily"]

    if len(daily) < TREND_FILTER_WINDOW:
        print(f"  بنجمع بيانات: {len(daily)}/{TREND_FILTER_WINDOW}  |  السعر: {price:.2f}")
        return None

    sma20 = float(np.mean(daily[-LOOKBACK_WINDOW:]))
    sma50 = float(np.mean(daily[-TREND_FILTER_WINDOW:]))

    trend_up = price > sma20 and price > sma50
    trend_broken = price < sma50

    status = "ترند صاعد" if trend_up else ("ترند منكسر" if trend_broken else "محايد")

    # أول مرة السهم يتضاف: تهيئة صامتة بدل ما كل سهم صاعد يبعت "بداية ترند"
    if not stock_state.get("initialized"):
        stock_state["last_signal"] = "BUY" if trend_up else "NONE"
        stock_state["initialized"] = True
        print(f"  [تهيئة] السعر: {price:.2f} | SMA20: {sma20:.2f} | SMA50: {sma50:.2f} | "
              f"الحالة: {status} - من غير تنبيه")
        return {"ticker": ticker, "signal": "INIT", "status": status}

    alert_signal = None
    if trend_up and stock_state["last_signal"] != "BUY":
        alert_signal = "BUY"
        stock_state["last_signal"] = "BUY"
    elif trend_broken and stock_state["last_signal"] == "BUY":
        alert_signal = "SELL"
        stock_state["last_signal"] = "SELL"

    print(f"  السعر: {price:.2f} | SMA20: {sma20:.2f} | SMA50: {sma50:.2f} | "
          f"الحالة: {status} | آخر إشارة: {stock_state['last_signal']}")

    if alert_signal:
        explanation = explain_alert(alert_signal, price, sma20, sma50)
        append_alert_log(ticker, alert_signal, price, sma20, sma50)
        outlook = compute_outlook(daily, price, trend_up, trend_broken)
        return {"ticker": ticker, "signal": alert_signal, "price": price,
                "sma20": sma20, "sma50": sma50, "explanation": explanation,
                "outlook": outlook}
    return None


def format_outlook(o):
    if not o:
        return ""
    return (
        f"🔮 توقع الجلسة الجاية: {o['label']}\n"
        f"احتمال صعود {o['p_up']:.0f}%  |  هبوط {o['p_down']:.0f}%\n"
        f"✅ تأكيد الصعود فنيًا: {o['tech']:.0f}% ({o['tech_k']} من 5 مؤشرات)\n"
        f"لو صعد: السعر في حدود {o['up_low']:.2f} - {o['up_high']:.2f} جنيه\n"
        f"لو هبط: السعر في حدود {o['down_low']:.2f} - {o['down_high']:.2f} جنيه\n"
        f"(تقدير إحصائي من آخر {o['days']} يوم، مش ضمان)\n\n"
    )


def forecast_table(state):
    rows = []
    for t in TICKERS:
        daily = state["stocks"][t]["daily"]
        if len(daily) < TREND_FILTER_WINDOW:
            continue
        price = daily[-1]
        o = compute_outlook(daily, price, False, False)
        if o:
            rows.append((o["tech"], o["p_up"], t.replace(".CA", ""), o))
    if not rows:
        return ""
    rows.sort(key=lambda r: (-r[0], -r[1]))
    lines = ["🔮 توقع الجلسة الجاية - كل الأسهم",
             "(مرتبة من الأعلى تأكيدًا للصعود)",
             "الاسم: اتجاه | صعود% | تأكيد% | لو صعد | لو هبط", ""]
    for tech, p_up, name, o in rows:
        arrow = "📈" if tech >= 60 else ("📉" if tech <= 40 else "➖")
        lines.append(f"{arrow} {name}: {p_up:.0f}% | {tech:.0f}% | "
                     f"{o['up_low']:.2f}-{o['up_high']:.2f} | "
                     f"{o['down_low']:.2f}-{o['down_high']:.2f}")
    lines += ["", "⚠️ تقديرات إحصائية من تاريخ كل سهم، مش توصية ولا ضمان."]
    return "\n".join(lines)


# ============================================================
# أوامر تليجرام: اكتب اسم سهم (مثلاً ABUK) والبوت يرد بتحليله.
# بيتعالج مع كل run (مش لحظي) ومن chat_id بتاعك بس.
# ============================================================
def analyze_stock_text(ticker, stock_state):
    name = ticker.replace(".CA", "")
    daily = stock_state.get("daily", [])
    dates = stock_state.get("dates", [])
    if len(daily) < TREND_FILTER_WINDOW:
        return (f"📊 {name}\nلسه بيجمع بيانات ({len(daily)}/{TREND_FILTER_WINDOW} يوم) "
                f"- مفيش تحليل كفاية دلوقتي.")
    price = daily[-1]
    sma20 = float(np.mean(daily[-LOOKBACK_WINDOW:]))
    sma50 = float(np.mean(daily[-TREND_FILTER_WINDOW:]))
    trend_up = price > sma20 and price > sma50
    trend_broken = price < sma50
    status = "ترند صاعد 📈" if trend_up else ("ترند منكسر 📉" if trend_broken else "محايد ➖")

    def pct(a, b):
        return (a / b - 1) * 100

    chg5 = pct(price, daily[-6]) if len(daily) > 5 else 0.0
    chg20 = pct(price, daily[-21]) if len(daily) > 20 else 0.0
    last = {"BUY": "شراء (الترند صاعد من وقتها)", "SELL": "بيع (الترند اتكسر)",
            "NONE": "مفيش إشارة لسه"}.get(stock_state.get("last_signal", "NONE"), "-")
    date = dates[-1] if dates else "-"
    outlook = compute_outlook(daily, price, trend_up, trend_broken)
    return (
        f"📊 {name} - تحليل\n"
        f"آخر سعر محفوظ: {price:.2f} جنيه ({date})\n"
        f"الحالة: {status}\n"
        f"المتوسط 20 يوم: {sma20:.2f} (السعر {pct(price, sma20):+.1f}%)\n"
        f"المتوسط 50 يوم: {sma50:.2f} (السعر {pct(price, sma50):+.1f}%)\n"
        f"التغير: 5 أيام {chg5:+.1f}%  |  20 يوم {chg20:+.1f}%\n"
        f"آخر إشارة: {last}\n\n"
        f"{format_outlook(outlook)}"
        f"⚠️ تحليل فني بسيط مبني على القفلات اليومية، مش توصية."
    )


def overview_text(state):
    groups = {"ترند صاعد": [], "ترند منكسر": [], "محايد": [], "بيجمع بيانات": []}
    for t in TICKERS:
        daily = state["stocks"][t]["daily"]
        name = t.replace(".CA", "")
        if len(daily) < TREND_FILTER_WINDOW:
            groups["بيجمع بيانات"].append(name)
            continue
        price = daily[-1]
        sma20 = float(np.mean(daily[-LOOKBACK_WINDOW:]))
        sma50 = float(np.mean(daily[-TREND_FILTER_WINDOW:]))
        if price > sma20 and price > sma50:
            groups["ترند صاعد"].append(name)
        elif price < sma50:
            groups["ترند منكسر"].append(name)
        else:
            groups["محايد"].append(name)
    icons = {"ترند صاعد": "📈", "ترند منكسر": "📉", "محايد": "➖", "بيجمع بيانات": "⏳"}
    lines = [f"{icons[k]} {k} ({len(v)}): {', '.join(v)}" for k, v in groups.items() if v]
    return "📋 حالة كل الأسهم\n" + "\n".join(lines)


def handle_telegram_commands(state):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import re
    meta = state.setdefault("meta", {})
    offset = meta.get("tg_offset", 0)
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
            params={"offset": offset, "timeout": 0}, timeout=15)
        updates = resp.json().get("result", []) if resp.status_code == 200 else []
    except Exception as e:
        print("[تليجرام] فشل قراءة الرسايل:", e)
        return

    bases = {t.replace(".CA", ""): t for t in TICKERS}
    for u in updates:
        meta["tg_offset"] = u["update_id"] + 1
        msg = u.get("message") or {}
        # بنرد على صاحب البوت بس
        if str(msg.get("chat", {}).get("id", "")) != str(TELEGRAM_CHAT_ID):
            continue
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        words = [w.upper() for w in re.findall(r"[A-Za-z]{3,6}", text)]
        if any(w in ("ALL", "LIST") for w in words) or text in ("/all", "/list", "الكل"):
            send_telegram(overview_text(state))
            continue
        ticker = next((bases[w] for w in words if w in bases), None)
        if ticker:
            send_telegram(analyze_stock_text(ticker, state["stocks"][ticker]))
        else:
            send_telegram("مش لاقي السهم ده في القايمة. اكتب كود السهم زي ABUK أو COMI، "
                          "أو all عشان تشوف حالة كل الأسهم.\n\nالأكواد: " + ", ".join(bases))


# ============================================================
# التقرير اليومي التلقائي (من غير ما تكتب حاجة)
# ============================================================
def send_long(text, limit=3800):
    chunk = ""
    for line in text.split("\n"):
        if len(chunk) + len(line) + 1 > limit and chunk:
            send_telegram(chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        send_telegram(chunk)


def build_daily_report(state, today):
    up = down = flat = 0
    movers = []          # (chg%, name, price)
    near_buy, near_sell = [], []
    counts = {"صاعد": 0, "منكسر": 0, "محايد": 0, "بيجمع": 0}

    for t in TICKERS:
        st = state["stocks"][t]
        daily, dates = st["daily"], st["dates"]
        name = t.replace(".CA", "")
        if len(daily) < TREND_FILTER_WINDOW:
            counts["بيجمع"] += 1
            continue
        price = daily[-1]
        sma20 = float(np.mean(daily[-LOOKBACK_WINDOW:]))
        sma50 = float(np.mean(daily[-TREND_FILTER_WINDOW:]))
        trend_up = price > sma20 and price > sma50
        trend_broken = price < sma50
        counts["صاعد" if trend_up else ("منكسر" if trend_broken else "محايد")] += 1

        fresh = bool(dates) and dates[-1] == today and len(daily) >= 2
        if fresh:
            chg = (price / daily[-2] - 1) * 100
            movers.append((chg, name, price))
            if chg > 0.005:
                up += 1
            elif chg < -0.005:
                down += 1
            else:
                flat += 1

        if not trend_up:
            gap = (max(sma20, sma50) / price - 1) * 100
            if 0 < gap <= NEAR_SIGNAL_PCT:
                near_buy.append((gap, name))
        elif st.get("last_signal") == "BUY":
            gap = (price / sma50 - 1) * 100
            if 0 <= gap <= NEAR_SIGNAL_PCT:
                near_sell.append((gap, name))

    movers.sort(reverse=True)
    lines = [f"📊 تقرير نهاية اليوم - {today}", ""]
    if movers:
        lines.append(f"السوق النهاردة: 🔺 {up} سهم طلع | 🔻 {down} نزل | ➖ {flat} ثابت")
    lines.append(f"📈 ترند صاعد: {counts['صاعد']}  |  📉 منكسر: {counts['منكسر']}  |  "
                 f"➖ محايد: {counts['محايد']}  |  ⏳ بيجمع داتا: {counts['بيجمع']}")

    if movers:
        top = [m for m in movers if m[0] > 0][:TOP_N]
        bottom = [m for m in reversed(movers) if m[0] < 0][:TOP_N]
        if top:
            lines += ["", "🔺 أكبر صعود النهاردة:"]
            lines += [f"{n} {c:+.1f}% ({p:.2f})" for c, n, p in top]
        if bottom:
            lines += ["", "🔻 أكبر هبوط النهاردة:"]
            lines += [f"{n} {c:+.1f}% ({p:.2f})" for c, n, p in bottom]

    if near_buy:
        near_buy.sort()
        lines += ["", f"🎯 قريب من بداية ترند صاعد (أقل من {NEAR_SIGNAL_PCT:.0f}% تحت المتوسط):",
                  ", ".join(f"{n} (-{g:.1f}%)" for g, n in near_buy[:10])]
    if near_sell:
        near_sell.sort()
        lines += ["", f"⚠️ قريب من كسر الترند (أقل من {NEAR_SIGNAL_PCT:.0f}% فوق SMA50):",
                  ", ".join(f"{n} (+{g:.1f}%)" for g, n in near_sell[:10])]

    focus = [t for t in DAILY_FOCUS if t in state["stocks"]]
    if focus:
        lines += ["", "📌 الأسهم اللي بتتابعها:"]
        for t in focus:
            lines += ["", analyze_stock_text(t, state["stocks"][t])]
    return "\n".join(lines)


def maybe_send_daily_report(state):
    cairo = datetime.now(ZoneInfo("Africa/Cairo"))
    today = cairo.strftime("%Y-%m-%d")
    meta = state.setdefault("meta", {})
    if not FORCE_RUN:
        if cairo.weekday() not in {6, 0, 1, 2, 3}:      # أيام التداول بس
            return
        if cairo.hour < DAILY_REPORT_HOUR:
            return
        if meta.get("report_date") == today:
            return
        # لو مفيش ولا سهم اتحدّث النهاردة (إجازة مثلاً) متبعتش تقرير قديم
        if not any(state["stocks"][t]["dates"] and state["stocks"][t]["dates"][-1] == today
                   for t in TICKERS):
            return
    send_long(build_daily_report(state, today))
    table = forecast_table(state)
    if table:
        send_long(table)
    if not FORCE_RUN:
        meta["report_date"] = today


# ============================================================
# MAIN
# ============================================================
def main():
    state = load_state()
    handle_telegram_commands(state)  # الرد على أسئلتك (شغال حتى والسوق مقفول)

    if not is_market_open_now() and not FORCE_RUN:
        print("السوق مقفول دلوقتي (برة مواعيد EGX) - مفيش فحص")
        maybe_send_daily_report(state)  # لو الـ run جه بعد القفل وفيه تقرير النهاردة لسه
        save_state(state)
        return

    results = []

    for ticker in TICKERS:
        print(f"\n[{ticker}]")
        result = process_stock(ticker, state["stocks"][ticker])
        if result:
            results.append(result)
        time.sleep(1)  # تهدئة بسيطة بين الطلبات (31 سهم)

    alerts = [r for r in results if r["signal"] != "INIT"]
    inits = [r for r in results if r["signal"] == "INIT"]

    # رسالة واحدة بس للأسهم الجديدة بدل رسالة لكل سهم
    if inits:
        def names(status):
            return ", ".join(r["ticker"].replace(".CA", "") for r in inits if r["status"] == status) or "-"
        send_telegram(
            f"✅ اتضاف {len(inits)} سهم للمراقبة (من غير تنبيهات أولية)\n"
            f"📈 ترند صاعد دلوقتي: {names('ترند صاعد')}\n"
            f"📉 ترند منكسر: {names('ترند منكسر')}\n"
            f"➖ محايد: {names('محايد')}\n\n"
            f"من هنا هتوصلك تنبيهات بس لما الإشارة تتغير."
        )

    # أسهم لسه ماجمعتش 50 يوم (مثلاً yfinance مش مغطيها) - تحذير مرة واحدة
    meta = state.setdefault("meta", {})
    warned = set(meta.get("short_warned", []))
    short = [t for t in TICKERS
             if len(state["stocks"][t]["daily"]) < TREND_FILTER_WINDOW and t not in warned]
    if short:
        send_telegram("⚠️ الأسهم دي مفيش ليها تاريخ كفاية من yfinance، هتتأخر لحد ما تجمع 50 يوم: "
                      + ", ".join(t.replace(".CA", "") for t in short))
        meta["short_warned"] = sorted(warned | set(short))

    if len(alerts) > ALERT_DIGEST_THRESHOLD:
        # أسهم كتير اتغيرت مرة واحدة: رسالة واحدة مختصرة بدل رسايل كتير
        lines = []
        for a in alerts:
            icon = "📈" if a["signal"] == "BUY" else "📉"
            label = "بداية ترند صاعد" if a["signal"] == "BUY" else "الترند اتكسر"
            lines.append(f"{icon} {a['ticker'].replace('.CA', '')} - {label} | "
                         f"{a['price']:.2f} (SMA50: {a['sma50']:.2f})")
        send_telegram(f"🔔 {len(alerts)} إشارة جديدة\n\n" + "\n".join(lines) +
                      "\n\nاكتب كود أي سهم (زي ABUK) عشان تاخد تحليله كامل.\n"
                      "⚠️ ده تنبيه بس - القرار ليك.")
        alerts_to_send = []
    else:
        alerts_to_send = alerts

    for a in alerts_to_send:
        icon = "📈" if a["signal"] == "BUY" else "📉"
        label = "بداية ترند صاعد" if a["signal"] == "BUY" else "الترند اتكسر"
        message = (
            f"{icon} {a['ticker'].replace('.CA', '')} - {label}\n"
            f"السعر: {a['price']:.2f} جنيه\n"
            f"المتوسط 20 يوم: {a['sma20']:.2f}  |  المتوسط 50 يوم: {a['sma50']:.2f}\n\n"
            f"{a['explanation']}\n\n"
            f"{format_outlook(a.get('outlook'))}"
            f"⚠️ ده تنبيه بس - القرار وتنفيذه في حسابك الحقيقي ليك."
        )
        send_telegram(message)

    if not alerts:
        print("\nمفيش تغيير في أي إشارة النهاردة - مفيش تنبيهات.")

    maybe_send_daily_report(state)
    save_state(state)


if __name__ == "__main__":
    main()
