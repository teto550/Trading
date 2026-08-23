"""
================================================================
   Live Stock Monitor - نسخة GitHub Actions (بدون جهاز شغال)
================================================================
السكريبت ده بيتشغّل مرة واحدة بس في كل مرة، ومحفظة وهمية (paper
portfolio) بتتبع من غير فلوس حقيقية عشان تتعلم إزاي الاستراتيجية
كانت هتتصرف. المحفظة بتبدأ بمبلغ افتراضي (100,000 جنيه)، وبتشتري
وتبيع أسهم حقيقية العدد بناءً على السعر الفعلي - من غير بيع مكشوف
(short selling)، لأن ده مش متاح عادة للمستثمر الفردي في EGX.
================================================================
"""

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import requests

TICKER = os.environ.get("TICKER", "COMI.CA")
LOOKBACK_WINDOW = 20
MEAN_REVERSION_STD = 1.5
STOP_LOSS_PCT = 0.03
STARTING_CASH = 100000.0  # رأس المال الافتراضي بالجنيه المصري
STATE_FILE = "state.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FORCE_RUN = os.environ.get("FORCE_RUN", "false").lower() == "true"


# ============================================================
# مواعيد تداول EGX (الأحد - الخميس، 10:00 ص - 2:30 م بتوقيت القاهرة)
# ============================================================
def is_market_open_now():
    cairo = datetime.now(ZoneInfo("Africa/Cairo"))
    trading_days = {6, 0, 1, 2, 3}  # Sunday, Monday, Tuesday, Wednesday, Thursday
    if cairo.weekday() not in trading_days:
        return False
    market_open = cairo.replace(hour=10, minute=0, second=0, microsecond=0)
    market_close = cairo.replace(hour=14, minute=30, second=0, microsecond=0)
    return market_open <= cairo <= market_close


# ============================================================
# الحالة المحفوظة (state.json)
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {}

    state.setdefault("prices", [])
    state.setdefault("cash", STARTING_CASH)
    state.setdefault("shares", 0)
    state.setdefault("avg_buy_price", None)
    return state


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# جلب السعر
# ============================================================
def fetch_price_mubasher():
    """
    بيجيب السعر من صفحة Mubasher العامة (متأخر 15 دقيقة وقت التداول،
    لكن بيتحدث فعلياً - عكس مشكلة Yahoo Finance الراكدة لسهم COMI).
    مصدر غير رسمي، ممكن يتعطل لو الموقع غيّر شكل صفحته.
    """
    import re
    symbol = TICKER.replace(".CA", "")
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


def fetch_price_yfinance():
    import yfinance as yf
    tk = yf.Ticker(TICKER)
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


def fetch_price():
    try:
        return fetch_price_mubasher()
    except Exception as e:
        print(f"[Mubasher] فشل الجلب ({e})، هجرب yfinance كـ fallback")
        return fetch_price_yfinance()


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


def fallback_explanation(action, price, sma, zscore):
    if action == "BUY":
        return f"السعر ({price:.2f}) فوق المتوسط ({sma:.2f})، إشارة اتجاه صاعد."
    if action == "SELL_SIGNAL":
        return "الإشارة رجعت محايدة أو هابطة، فتم البيع."
    if action == "SELL_STOP_LOSS":
        return "تم البيع تلقائياً بسبب وصول الخسارة لحد وقف الخسارة."
    return f"Z-score الحالي {zscore:+.2f} هو اللي حرّك القرار."


def explain_trade(action, price, sma, zscore, extra_context):
    if not GEMINI_API_KEY:
        return fallback_explanation(action, price, sma, zscore)
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""اكتب جملتين بس بالعربي البسيط (مصري) تشرح لمستثمر عادي
ليه حصل الإجراء ده في التداول، من غير مقدمة ولا خاتمة:

الإجراء: {action}
السعر الحالي: {price:.2f}
المتوسط المتحرك: {sma:.2f}
Z-score: {zscore:.2f}
{extra_context}
"""
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text if text else fallback_explanation(action, price, sma, zscore)
    except Exception as e:
        print("[Gemini] فشل، هستخدم شرح تلقائي:", e)
        return fallback_explanation(action, price, sma, zscore)


# ============================================================
# MAIN - تشغيلة واحدة
# ============================================================
def main():
    if not is_market_open_now() and not FORCE_RUN:
        print("السوق مقفول دلوقتي (برة مواعيد EGX) - مفيش تنفيذ")
        return

    state = load_state()

    try:
        price = fetch_price()
    except Exception as e:
        print("خطأ في جلب السعر:", e)
        return

    state["prices"].append(price)
    state["prices"] = state["prices"][-LOOKBACK_WINDOW:]

    portfolio_value = state["cash"] + state["shares"] * price

    if len(state["prices"]) < LOOKBACK_WINDOW:
        print(f"بنجمع بيانات كفاية: {len(state['prices'])}/{LOOKBACK_WINDOW}  |  "
              f"السعر الحالي: {price:.2f}  |  قيمة المحفظة: {portfolio_value:,.2f} جنيه")
        save_state(state)
        return

    arr = np.array(state["prices"])
    sma = arr.mean()
    std = arr.std()

    momentum_signal = 1 if price > sma else -1
    zscore = (price - sma) / std if std > 0 else 0
    if zscore > MEAN_REVERSION_STD:
        mr_signal = -1
    elif zscore < -MEAN_REVERSION_STD:
        mr_signal = 1
    else:
        mr_signal = 0

    raw_score = 0.6 * momentum_signal + 0.4 * mr_signal
    if raw_score > 0.15:
        signal = 1
    elif raw_score < -0.15:
        signal = -1
    else:
        signal = 0

    action = None
    message_extra = ""

    # وقف الخسارة - له الأولوية قبل أي حاجة تانية
    if state["shares"] > 0 and state["avg_buy_price"]:
        change_pct = (price - state["avg_buy_price"]) / state["avg_buy_price"]
        if change_pct < -STOP_LOSS_PCT:
            proceeds = state["shares"] * price
            profit = proceeds - (state["shares"] * state["avg_buy_price"])
            sold_shares = state["shares"]
            state["cash"] += proceeds
            state["shares"] = 0
            state["avg_buy_price"] = None
            action = "SELL_STOP_LOSS"
            message_extra = (
                f"باع {sold_shares} سهم بسعر {price:.2f} جنيه "
                f"(إجمالي {proceeds:,.2f} جنيه)\n"
                f"الخسارة: {profit:,.2f} جنيه ({change_pct:+.2%})\n"
                f"الكاش دلوقتي: {state['cash']:,.2f} جنيه"
            )

    # لو معندناش أسهم ولسه معندناش action من وقف الخسارة، واتولدت إشارة شراء
    if action is None and state["shares"] == 0 and signal == 1:
        shares_to_buy = int(state["cash"] // price)
        if shares_to_buy > 0:
            cost = shares_to_buy * price
            state["cash"] -= cost
            state["shares"] = shares_to_buy
            state["avg_buy_price"] = price
            action = "BUY"
            message_extra = (
                f"اشترى {shares_to_buy} سهم بسعر {price:.2f} جنيه "
                f"(إجمالي {cost:,.2f} جنيه)\n"
                f"باقي الكاش: {state['cash']:,.2f} جنيه"
            )

    # لو عندنا أسهم والإشارة بقت محايدة أو هابطة -> بيع
    elif action is None and state["shares"] > 0 and signal <= 0:
        proceeds = state["shares"] * price
        profit = proceeds - (state["shares"] * state["avg_buy_price"])
        profit_pct = profit / (state["shares"] * state["avg_buy_price"])
        sold_shares = state["shares"]
        state["cash"] += proceeds
        state["shares"] = 0
        state["avg_buy_price"] = None
        action = "SELL_SIGNAL"
        message_extra = (
            f"باع {sold_shares} سهم بسعر {price:.2f} جنيه "
            f"(إجمالي {proceeds:,.2f} جنيه)\n"
            f"{'ربح' if profit >= 0 else 'خسارة'}: {profit:,.2f} جنيه ({profit_pct:+.2%})\n"
            f"الكاش دلوقتي: {state['cash']:,.2f} جنيه"
        )

    portfolio_value = state["cash"] + state["shares"] * price
    total_return_pct = (portfolio_value - STARTING_CASH) / STARTING_CASH

    holding_text = f"{state['shares']} سهم" if state["shares"] > 0 else "كاش بالكامل"
    print(f"السعر: {price:.2f} | SMA: {sma:.2f} | Z: {zscore:+.2f} | "
          f"المحفظة: {holding_text} | القيمة الكلية: {portfolio_value:,.2f} جنيه "
          f"({total_return_pct:+.2%})")

    if action:
        explanation = explain_trade(action, price, sma, zscore, message_extra)
        print(f">>> إجراء: {action}\n{message_extra}\nالشرح: {explanation}")
        message = (
            f"📈 {TICKER}\n"
            f"{message_extra}\n\n"
            f"{explanation}\n\n"
            f"قيمة المحفظة: {portfolio_value:,.2f} جنيه ({total_return_pct:+.2%})"
        )
        send_telegram(message)

    save_state(state)


if __name__ == "__main__":
    main()
