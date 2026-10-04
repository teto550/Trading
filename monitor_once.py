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
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import requests

TICKERS = [t.strip() for t in os.environ.get(
    # GGRN اتشال: مش مغطى في yfinance/Yahoo Finance خالص
    "TICKERS", "COMI.CA,EFIH.CA,EFID.CA,ABUK.CA"
).split(",") if t.strip()]

LOOKBACK_WINDOW = 20
TREND_FILTER_WINDOW = 50
# بنحتفظ بقفلات يومية (مش قراءات كل 15 دقيقة) - 60 يوم كفاية للـ SMA50
HISTORY_WINDOW = 60
STATE_FILE = "state.json"
ALERT_LOG_FILE = "alert_log.csv"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
FORCE_RUN = os.environ.get("FORCE_RUN", "false").lower() == "true"

# لو السعر الجديد مختلف عن آخر سعر معروف بنسبة أكبر من الحد ده،
# هنتأكد منه عن طريق yfinance قبل ما نصدقه
PRICE_SANITY_THRESHOLD = 0.20


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
    return {"dates": [], "daily": [], "last_signal": "NONE"}  # NONE / BUY / SELL


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
    return {"stocks": stocks}


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


def seed_history(ticker, stock_state):
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="6mo", interval="1d")["Close"].dropna()
    if h.empty:
        raise ValueError("مفيش تاريخ متاح من yfinance")
    h = h.tail(HISTORY_WINDOW)
    stock_state["daily"] = [float(x) for x in h]
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
    if n < 20:
        return None
    lo, hi = np.percentile(rets, [10, 90])  # ~80% من الأيام وقعت جوه المدى ده
    if trend_up:
        label = "صاعد 📈"
    elif trend_broken:
        label = "هابط 📉"
    else:
        label = "محايد ➖"
    return {
        "label": label,
        "range_low": price * (1 + lo),
        "range_high": price * (1 + hi),
        "p_up": float((rets > 0).mean() * 100),
        "p_down": float((rets < 0).mean() * 100),
        "days": n,
    }


# ============================================================
# معالجة سهم واحد - بس بيحسب الإشارة وبيقارنها بآخر إشارة معروفة،
# مفيش تنفيذ ولا محفظة خالص
# ============================================================
def process_stock(ticker, stock_state):
    if not stock_state.get("daily"):
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

    # رفض القفزة الغريبة: متتخزنش ويتبعتلك تحذير تراجعه يدويًا
    if prev_price and abs(price - prev_price) / prev_price > PRICE_SANITY_THRESHOLD:
        print(f"  [رفض] قفزة غريبة {prev_price:.2f} -> {price:.2f}، اتجاهلت")
        send_telegram(f"⚠️ {ticker}: قفزة سعر غريبة {prev_price:.2f} → {price:.2f}، "
                      f"اتجاهلت ومتخزنتش. راجعها يدويًا.")
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

    alert_signal = None
    if trend_up and stock_state["last_signal"] != "BUY":
        alert_signal = "BUY"
        stock_state["last_signal"] = "BUY"
    elif trend_broken and stock_state["last_signal"] == "BUY":
        alert_signal = "SELL"
        stock_state["last_signal"] = "SELL"

    status = "ترند صاعد" if trend_up else ("ترند منكسر" if trend_broken else "محايد")
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
        f"🔮 مؤشر الجلسة الجاية: {o['label']}\n"
        f"المدى المتوقع (احتمال حوالي 80%): {o['range_low']:.2f} - {o['range_high']:.2f} جنيه\n"
        f"احتمال القفلة أعلى من النهاردة: {o['p_up']:.0f}%  |  أقل: {o['p_down']:.0f}%\n"
        f"(تقدير إحصائي من آخر {o['days']} يوم، مش ضمان)\n\n"
    )


# ============================================================
# MAIN
# ============================================================
def main():
    if not is_market_open_now() and not FORCE_RUN:
        print("السوق مقفول دلوقتي (برة مواعيد EGX) - مفيش فحص")
        return

    state = load_state()
    alerts = []

    for ticker in TICKERS:
        print(f"\n[{ticker}]")
        result = process_stock(ticker, state["stocks"][ticker])
        if result:
            alerts.append(result)

    for a in alerts:
        icon = "📈" if a["signal"] == "BUY" else "📉"
        label = "بداية ترند صاعد" if a["signal"] == "BUY" else "الترند اتكسر"
        message = (
            f"{icon} {a['ticker']} - {label}\n"
            f"السعر: {a['price']:.2f} جنيه\n"
            f"المتوسط 20 يوم: {a['sma20']:.2f}  |  المتوسط 50 يوم: {a['sma50']:.2f}\n\n"
            f"{a['explanation']}\n\n"
            f"{format_outlook(a.get('outlook'))}"
            f"⚠️ ده تنبيه بس - القرار وتنفيذه في حسابك الحقيقي ليك."
        )
        send_telegram(message)

    if not alerts:
        print("\nمفيش تغيير في أي إشارة النهاردة - مفيش تنبيهات.")

    save_state(state)


if __name__ == "__main__":
    main()
