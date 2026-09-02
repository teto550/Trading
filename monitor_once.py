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
HISTORY_WINDOW = TREND_FILTER_WINDOW
STATE_FILE = "state.json"
ALERT_LOG_FILE = "alert_log.csv"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
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
# الحالة المحفوظة - بس تاريخ الأسعار وآخر إشارة اتبعتت، من غير أي
# محفظة أو كاش
# ============================================================
def default_stock_state():
    return {"prices": [], "last_signal": "NONE"}  # NONE / BUY / SELL


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
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""اكتب جملتين بس بالعربي البسيط (مصري) تشرح لمستثمر عادي
ليه ظهرت الإشارة دي، من غير مقدمة ولا خاتمة:

الإشارة: {"بداية ترند صاعد" if signal == "BUY" else "انكسار الترند الصاعد"}
السعر الحالي: {price:.2f}
المتوسط القريب (20 يوم): {sma20:.2f}
المتوسط الطويل (50 يوم): {sma50:.2f}
"""
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text if text else fallback_explanation(signal, price, sma20, sma50)
    except Exception as e:
        print("  [Gemini] فشل، هستخدم شرح تلقائي:", e)
        return fallback_explanation(signal, price, sma20, sma50)


# ============================================================
# معالجة سهم واحد - بس بيحسب الإشارة وبيقارنها بآخر إشارة معروفة،
# مفيش تنفيذ ولا محفظة خالص
# ============================================================
def process_stock(ticker, stock_state):
    prev_price = stock_state["prices"][-1] if stock_state["prices"] else None
    try:
        price = fetch_price(ticker, prev_price=prev_price)
    except Exception as e:
        print(f"  خطأ في جلب السعر: {e}")
        return None

    history = stock_state["prices"]
    if len(history) < TREND_FILTER_WINDOW:
        history.append(price)
        stock_state["prices"] = history[-HISTORY_WINDOW:]
        print(f"  بنجمع بيانات: {len(stock_state['prices'])}/{TREND_FILTER_WINDOW}  |  السعر: {price:.2f}")
        return None

    sma20 = np.array(history[-LOOKBACK_WINDOW:]).mean()
    sma50 = np.array(history[-TREND_FILTER_WINDOW:]).mean()

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

    stock_state["prices"].append(price)
    stock_state["prices"] = stock_state["prices"][-HISTORY_WINDOW:]

    if alert_signal:
        explanation = explain_alert(alert_signal, price, sma20, sma50)
        append_alert_log(ticker, alert_signal, price, sma20, sma50)
        return {"ticker": ticker, "signal": alert_signal, "price": price,
                "sma20": sma20, "sma50": sma50, "explanation": explanation}
    return None


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
            f"⚠️ ده تنبيه بس - القرار وتنفيذه في حسابك الحقيقي ليك."
        )
        send_telegram(message)

    if not alerts:
        print("\nمفيش تغيير في أي إشارة النهاردة - مفيش تنبيهات.")

    save_state(state)


if __name__ == "__main__":
    main()
