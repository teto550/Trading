"""
================================================================
   Live Stock Monitor - نسخة متعددة الأسهم مع إدارة مخاطر ذكية
================================================================
بيراقب أكتر من سهم في نفس الوقت، وبيوزع رأس المال بالتساوي بينهم
(تنويع). كل سهم عنده وقف خسارة وهدف ربح محسوبين من تقلبه هو نفسه
(مش نسبة ثابتة). كمان عنده "درجة ثقة" بتقل بعد خسارة وترجع تزيد
تدريجياً بعد كسب - أسلوب معروف لتقليل حجم الصفقة بعد أداء سيء،
مش "ذكاء اصطناعي" حقيقي بيتعلم، مهم نكون واضحين في ده.
================================================================
"""

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import requests

TICKERS = [t.strip() for t in os.environ.get(
    "TICKERS", "COMI.CA,GGRN.CA,EFIH.CA,EFID.CA,ABUK.CA"
).split(",") if t.strip()]

LOOKBACK_WINDOW = 20
MEAN_REVERSION_STD = 1.5
STARTING_CASH_TOTAL = 100000.0
MIN_CONFIDENCE = 0.3   # أقل نسبة من الخزنة يقدر يستخدمها بعد خسائر متتالية
MAX_CONFIDENCE = 1.0
STATE_FILE = "state.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FORCE_RUN = os.environ.get("FORCE_RUN", "false").lower() == "true"


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
# الحالة المحفوظة
# ============================================================
def default_stock_state(cash_slot):
    return {
        "prices": [],
        "cash": cash_slot,
        "shares": 0,
        "avg_buy_price": None,
        "confidence": MAX_CONFIDENCE,
    }


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {}

    cash_slot = STARTING_CASH_TOTAL / len(TICKERS)
    stocks = raw.get("stocks", {})

    for ticker in TICKERS:
        if ticker not in stocks:
            stocks[ticker] = default_stock_state(cash_slot)
        else:
            stocks[ticker].setdefault("confidence", MAX_CONFIDENCE)

    return {"stocks": stocks}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# جلب السعر
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


def fetch_price(ticker):
    try:
        return fetch_price_mubasher(ticker)
    except Exception as e:
        print(f"  [Mubasher] فشل الجلب ({e})، هجرب yfinance كـ fallback")
        return fetch_price_yfinance(ticker)


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
    if action == "SELL_TAKE_PROFIT":
        return "تم البيع تلقائياً بعد الوصول لهدف الربح المحدد."
    return f"Z-score الحالي {zscore:+.2f} هو اللي حرّك القرار."


def explain_trade(action, price, sma, zscore, extra_context):
    if not GEMINI_API_KEY:
        return fallback_explanation(action, price, sma, zscore)
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""اكتب جملتين بس بالعربي البسيط (مصري) تشرح لمستثمر عادي
ليه حصل الإجراء ده، من غير مقدمة ولا خاتمة:

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
        print("  [Gemini] فشل، هستخدم شرح تلقائي:", e)
        return fallback_explanation(action, price, sma, zscore)


# ============================================================
# معالجة سهم واحد
# ============================================================
def process_stock(ticker, stock_state):
    try:
        price = fetch_price(ticker)
    except Exception as e:
        print(f"  خطأ في جلب السعر: {e}")
        return None

    stock_state["prices"].append(price)
    stock_state["prices"] = stock_state["prices"][-LOOKBACK_WINDOW:]

    value = stock_state["cash"] + stock_state["shares"] * price

    if len(stock_state["prices"]) < LOOKBACK_WINDOW:
        print(f"  بنجمع بيانات: {len(stock_state['prices'])}/{LOOKBACK_WINDOW}  |  "
              f"السعر: {price:.2f}  |  قيمة الخزنة: {value:,.2f} جنيه")
        return None

    arr = np.array(stock_state["prices"])
    sma = arr.mean()
    std = arr.std()

    # وقف خسارة وهدف ربح مبنيين على تقلب السهم نفسه (مش نسبة ثابتة)
    volatility_ratio = (std / sma) if sma > 0 else 0.02
    stop_loss_pct = min(max(2.5 * volatility_ratio, 0.02), 0.08)
    take_profit_pct = stop_loss_pct * 2  # نسبة مخاطرة:عائد 1:2

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
    extra = ""

    # وقف خسارة / هدف ربح - لهم الأولوية
    if stock_state["shares"] > 0 and stock_state["avg_buy_price"]:
        change_pct = (price - stock_state["avg_buy_price"]) / stock_state["avg_buy_price"]

        if change_pct < -stop_loss_pct:
            action = "SELL_STOP_LOSS"
        elif change_pct > take_profit_pct:
            action = "SELL_TAKE_PROFIT"

        if action:
            proceeds = stock_state["shares"] * price
            profit = proceeds - (stock_state["shares"] * stock_state["avg_buy_price"])
            profit_pct = profit / (stock_state["shares"] * stock_state["avg_buy_price"])
            sold_shares = stock_state["shares"]
            stock_state["cash"] += proceeds
            stock_state["shares"] = 0
            stock_state["avg_buy_price"] = None

            # تحديث درجة الثقة بناءً على النتيجة
            if profit >= 0:
                stock_state["confidence"] = min(MAX_CONFIDENCE, stock_state["confidence"] * 1.1)
            else:
                stock_state["confidence"] = max(MIN_CONFIDENCE, stock_state["confidence"] * 0.85)

            label = "هدف الربح" if action == "SELL_TAKE_PROFIT" else "وقف الخسارة"
            extra = (
                f"باع {sold_shares} سهم بسعر {price:.2f} جنيه (وصل {label})\n"
                f"إجمالي البيع: {proceeds:,.2f} جنيه\n"
                f"{'ربح' if profit >= 0 else 'خسارة'}: {profit:,.2f} جنيه ({profit_pct:+.2%})\n"
                f"كاش الخزنة دلوقتي: {stock_state['cash']:,.2f} جنيه"
            )

    # شراء
    if action is None and stock_state["shares"] == 0 and signal == 1:
        usable_cash = stock_state["cash"] * stock_state["confidence"]
        shares_to_buy = int(usable_cash // price)
        if shares_to_buy > 0:
            cost = shares_to_buy * price
            stock_state["cash"] -= cost
            stock_state["shares"] = shares_to_buy
            stock_state["avg_buy_price"] = price
            action = "BUY"
            extra = (
                f"اشترى {shares_to_buy} سهم بسعر {price:.2f} جنيه "
                f"(إجمالي {cost:,.2f} جنيه)\n"
                f"درجة الثقة الحالية: {stock_state['confidence']:.0%}\n"
                f"باقي كاش الخزنة: {stock_state['cash']:,.2f} جنيه"
            )

    # بيع بإشارة عادية (مش وقف خسارة ولا هدف ربح)
    elif action is None and stock_state["shares"] > 0 and signal <= 0:
        proceeds = stock_state["shares"] * price
        profit = proceeds - (stock_state["shares"] * stock_state["avg_buy_price"])
        profit_pct = profit / (stock_state["shares"] * stock_state["avg_buy_price"])
        sold_shares = stock_state["shares"]
        stock_state["cash"] += proceeds
        stock_state["shares"] = 0
        stock_state["avg_buy_price"] = None

        if profit >= 0:
            stock_state["confidence"] = min(MAX_CONFIDENCE, stock_state["confidence"] * 1.1)
        else:
            stock_state["confidence"] = max(MIN_CONFIDENCE, stock_state["confidence"] * 0.85)

        action = "SELL_SIGNAL"
        extra = (
            f"باع {sold_shares} سهم بسعر {price:.2f} جنيه (إشارة اتجاه)\n"
            f"إجمالي البيع: {proceeds:,.2f} جنيه\n"
            f"{'ربح' if profit >= 0 else 'خسارة'}: {profit:,.2f} جنيه ({profit_pct:+.2%})\n"
            f"كاش الخزنة دلوقتي: {stock_state['cash']:,.2f} جنيه"
        )

    value = stock_state["cash"] + stock_state["shares"] * price
    holding = f"{stock_state['shares']} سهم" if stock_state["shares"] > 0 else "كاش بالكامل"
    print(f"  السعر: {price:.2f} | SMA: {sma:.2f} | Z: {zscore:+.2f} | "
          f"وقف خسارة: {stop_loss_pct:.1%} | هدف ربح: {take_profit_pct:.1%} | "
          f"المحفظة: {holding} | القيمة: {value:,.2f} جنيه | ثقة: {stock_state['confidence']:.0%}")

    if action:
        explanation = explain_trade(action, price, sma, zscore, extra)
        print(f"  >>> إجراء: {action}\n  {extra}\n  الشرح: {explanation}")
        return {
            "ticker": ticker, "action": action, "extra": extra,
            "explanation": explanation, "value": value,
        }
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    if not is_market_open_now() and not FORCE_RUN:
        print("السوق مقفول دلوقتي (برة مواعيد EGX) - مفيش تنفيذ")
        return

    state = load_state()
    trade_events = []

    for ticker in TICKERS:
        print(f"\n[{ticker}]")
        result = process_stock(ticker, state["stocks"][ticker])
        if result:
            trade_events.append(result)

    total_value = sum(
        s["cash"] + s["shares"] * (s["prices"][-1] if s["prices"] else 0)
        for s in state["stocks"].values()
    )
    total_return_pct = (total_value - STARTING_CASH_TOTAL) / STARTING_CASH_TOTAL
    print(f"\nإجمالي قيمة المحفظة (كل الأسهم): {total_value:,.2f} جنيه ({total_return_pct:+.2%})")

    for event in trade_events:
        message = (
            f"📈 {event['ticker']}\n"
            f"{event['extra']}\n\n"
            f"{event['explanation']}\n\n"
            f"إجمالي المحفظة الكلية: {total_value:,.2f} جنيه ({total_return_pct:+.2%})"
        )
        send_telegram(message)

    save_state(state)


if __name__ == "__main__":
    main()
