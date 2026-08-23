"""
================================================================
   Live Stock Monitor - نسخة GitHub Actions (بدون جهاز شغال)
================================================================
السكريبت ده بيتشغّل مرة واحدة بس في كل مرة (مش loop لا نهائي)،
لأن GitHub Actions بيشغّل السكريبت على فترات بدل ما يسيبه شغال
طول الوقت. الحالة (آخر 20 سعر، المركز المفتوح...) بتتحفظ في
ملف state.json وبترجع تتقرأ في المرة الجاية.
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
STATE_FILE = "state.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FORCE_RUN = os.environ.get("FORCE_RUN", "false").lower() == "true"

POSITION_NAMES = {1: "شراء (Long)", -1: "بيع (Short)", 0: "خارج السوق (Flat)"}


# ============================================================
# مواعيد تداول EGX (الأحد - الخميس، 10:00 ص - 2:30 م بتوقيت القاهرة)
# ============================================================
def is_market_open_now():
    cairo = datetime.now(ZoneInfo("Africa/Cairo"))
    # Python weekday(): Monday=0 ... Sunday=6
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
            return json.load(f)
    return {"prices": [], "position": 0, "entry_price": None}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# جلب السعر
# ============================================================
def fetch_price():
    import yfinance as yf
    tk = yf.Ticker(TICKER)
    try:
        price = tk.fast_info["last_price"]
        if price is not None and price == price:  # يستبعد NaN
            return float(price)
    except Exception:
        pass
    data = tk.history(period="5d", interval="1d")
    if data.empty:
        raise ValueError("مفيش بيانات متاحة")
    return float(data.iloc[-1]["Close"])


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
    if action in ("OPEN_LONG", "REVERSE_POSITION") and price > sma:
        return f"السعر ({price:.2f}) فوق المتوسط ({sma:.2f})، إشارة اتجاه صاعد."
    if action == "OPEN_SHORT":
        return f"السعر ({price:.2f}) تحت المتوسط ({sma:.2f})، إشارة اتجاه هابط."
    if action == "STOP_LOSS_EXIT":
        return "تم الخروج تلقائياً من الصفقة بسبب وقف الخسارة."
    if action == "CLOSE_FLAT":
        return "تم إغلاق المركز لأن الإشارة رجعت محايدة."
    return f"Z-score الحالي {zscore:+.2f} هو اللي حرّك القرار."


def explain_trade(action, price, sma, zscore, position_label):
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
الموقف الجديد: {position_label}
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

    if len(state["prices"]) < LOOKBACK_WINDOW:
        print(f"بنجمع بيانات كفاية: {len(state['prices'])}/{LOOKBACK_WINDOW}  |  السعر الحالي: {price:.2f}")
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

    position = state["position"]
    entry_price = state["entry_price"]
    action = None

    if position != 0 and entry_price is not None:
        change_pct = (price - entry_price) / entry_price
        hit_stop = (
            (position > 0 and change_pct < -STOP_LOSS_PCT)
            or (position < 0 and change_pct > STOP_LOSS_PCT)
        )
        if hit_stop:
            action = "STOP_LOSS_EXIT"
            position = 0
            entry_price = None

    if position == 0 and signal != 0:
        position = signal
        entry_price = price
        action = "OPEN_LONG" if signal == 1 else "OPEN_SHORT"
    elif position != 0 and signal == 0:
        action = "CLOSE_FLAT"
        position = 0
        entry_price = None
    elif position != 0 and signal != 0 and np.sign(signal) != np.sign(position):
        action = "REVERSE_POSITION"
        position = signal
        entry_price = price

    print(f"السعر: {price:.2f} | SMA: {sma:.2f} | Z: {zscore:+.2f} | الموقف: {POSITION_NAMES[position]}")

    if action:
        explanation = explain_trade(action, price, sma, zscore, POSITION_NAMES[position])
        print(f">>> إجراء: {action}\nالشرح: {explanation}")
        message = f"📈 {TICKER}\nالإجراء: {action}\nالسعر: {price:.2f}\n\n{explanation}"
        send_telegram(message)

    state["position"] = position
    state["entry_price"] = entry_price
    save_state(state)


if __name__ == "__main__":
    main()
