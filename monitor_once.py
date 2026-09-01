"""
================================================================
   Live Stock Monitor - نسخة متعددة الأسهم مع إدارة مخاطر ذكية
================================================================
بيراقب أكتر من سهم في نفس الوقت، وبيوزع رأس المال بالتساوي بينهم
(تنويع). كل سهم عنده وقف خسارة وهدف ربح محسوبين من تقلبه هو نفسه
(مش نسبة ثابتة). كمان عنده "درجة ثقة" بتقل بعد خسارة وترجع تزيد
تدريجياً بعد كسب - أسلوب معروف لتقليل حجم الصفقة بعد أداء سيء،
مش "ذكاء اصطناعي" حقيقي بيتعلم، مهم نكون واضحين في ده.

تحديثات مهمة (v2):
- إصلاح فرق كان موجود بين اللايف والباكتست: الـ SMA/Z-score كانوا
  بيتحسبوا هنا من نافذة شاملة السعر الحالي نفسه، بينما الباكتست كان
  بيحسبهم من الأسعار اللي قبل السعر الحالي بس. دلوقتي الاتنين
  متطابقين (نافذة الـ 20 سعر اللي قبل السعر الحالي، والسعر الحالي
  بيتقارن بيها من بره).
- إضافة تكلفة تداول تقريبية (عمولة سمسرة + رسوم) على كل صفقة، عشان
  الأرباح المحسوبة تبقى أقرب للواقع. رقم التكلفة تقريبي - راجع
  عمولة السمسار بتاعك الفعلية وعدّل TRANSACTION_COST_PCT لو مختلفة.
- فحص أمان على السعر المجلوب من Mubasher: لو فيه قفزة كبيرة وغير
  منطقية (أكتر من 20% في 15 دقيقة) بيتأكد من yfinance كمصدر تاني
  قبل ما يثق في السعر ده.
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
    # GGRN اتشال من هنا: اتدرج على EGX في يوليو 2024 بس، بيتداول في
    # سوق النيل (NOPL) للشركات الناشئة، ومش مغطى في قاعدة بيانات
    # yfinance/Yahoo Finance خالص (مش مشكلة رمز، مفيش بيانات عنه أصلاً)
    "TICKERS", "COMI.CA,EFIH.CA,EFID.CA,ABUK.CA"
).split(",") if t.strip()]

# "hybrid" = النسخة الأصلية (momentum + mean-reversion مخلوطين)
# "momentum_only" = momentum بس، بدون trailing stop
# "momentum_trailing" = momentum بس + وقف خسارة متحرّك
# "trend_following" = فلتر ترند طويل المدى (50 يوم) + وقف خسارة متحرّك
# "trend_adx_sr" = [تجريبي] زي trend_following + فلتر قوة الترند
#     (ADX) + نقاط دعم/مقاومة. ملحوظة مهمة: اللايف بيجمع لقطات سعر كل
#     15 دقيقة (مش شمعات يومية كاملة فيها Open/High/Low)، فـ ADX هنا
#     نسخة *مبسّطة* (تقريب) من الحساب الكلاسيكي اللي في الباكتست
#     (اللي بيستخدم High/Low حقيقيين). جرّبها في الباكتست الأول
#     وتأكد من نتيجتها خارج العينة قبل ما تشغّلها هنا لايف.
#
# النتيجة دي الأفضل منهجيًا من بين كل النسخ اللي اتجرّبت (باكتست على
# عينتين منفصلتين من الأسهم، وطلعت موجبة في الاتنين - راجع نتائج
# backtest.py). لسه بتخسر قدام "اشتري واستني" في سوق صاعد بقوة زي
# اللي كان في الفترة دي، لكنها الأقل مخاطرة (أقل تراجع أقصى) والأكتر
# اتساقًا من كل النسخ التانية.
STRATEGY_MODE = os.environ.get("STRATEGY_MODE", "trend_following")

LOOKBACK_WINDOW = 20
TREND_FILTER_WINDOW = 50  # متوسط طويل المدى لفلتر الترند
SWING_WINDOW = 5          # نافذة تأكيد نقاط الدعم/المقاومة (أصغر من
                           # الباكتست لأن تاريخ اللايف محدود)
ADX_PERIOD = 14
ADX_THRESHOLD = 25        # حد قوة الترند الأدنى لدخول trend_adx_sr
SR_BUFFER_PCT = 0.02       # مسافة الأمان من المقاومة (trend_adx_sr)
# أكبر نافذة محتاجينها في الذاكرة - لازم نخزن تاريخ كفاية لكل الأوضاع
HISTORY_WINDOW = max(LOOKBACK_WINDOW, TREND_FILTER_WINDOW) + SWING_WINDOW
MEAN_REVERSION_STD = 1.5
# حد المنطقة المحايدة لـ raw_score: لازم يكون أكبر من (0.6 - 0.4 = 0.2)
# عشان لما الـ momentum والـ mean-reversion يتعارضوا فعلاً (يبقى الفرق
# 0.2) يدخلوا "منطقة محايدة" (مفيش تنفيذ) زي ما هي نية التصميم. كان
# الحد قديمًا 0.15 وده أقل من 0.2، فكانت المنطقة المحايدة دي عمرها ما
# بتتفعل - الإشارة كانت دايمًا بتاخد اتجاه الـ momentum بس، وكأن
# الـ mean-reversion مش موجود أصلاً.
SIGNAL_THRESHOLD = 0.25
STARTING_CASH_TOTAL = 100000.0
MIN_CONFIDENCE = 0.3   # أقل نسبة من الخزنة يقدر يستخدمها بعد خسائر متتالية
MAX_CONFIDENCE = 1.0
STATE_FILE = "state.json"
TRADE_LOG_FILE = "trade_log.csv"

# تكلفة تداول تقريبية (عمولة سمسرة + رسوم بورصة/تسوية + دمغة) كنسبة
# من قيمة الصفقة، بتتطبق على الشرا والبيع الاتنين. ده رقم تقريبي
# لتمثيل تكلفة واقعية - غيّره لو عمولة السمسار بتاعك مختلفة.
TRANSACTION_COST_PCT = 0.0035  # 0.35%

# لو السعر الجديد من Mubasher مختلف عن آخر سعر معروف بنسبة أكبر من
# الحد ده، هنتأكد منه عن طريق yfinance قبل ما نصدقه
PRICE_SANITY_THRESHOLD = 0.20  # 20%

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
def compute_adx_lite(prices, period=ADX_PERIOD):
    """
    نسخة مبسّطة (تقريب) من ADX باستخدام سعر الإغلاق بس - من غير
    High/Low حقيقيين (اللايف بيجمع لقطات سعر مش شمعات كاملة). النتيجة
    تقريب لقوة الترند، مش ADX الكلاسيكي بالظبط زي الباكتست.
    """
    if len(prices) < period + 1:
        return None
    diffs = np.diff(np.array(prices[-(period + 1):]))
    plus_dm = np.where(diffs > 0, diffs, 0.0)
    minus_dm = np.where(diffs < 0, -diffs, 0.0)
    tr = np.abs(diffs)
    atr = tr.mean()
    if atr == 0:
        return 0.0
    plus_di = 100 * plus_dm.mean() / atr
    minus_di = 100 * minus_dm.mean() / atr
    di_sum = plus_di + minus_di
    return 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0


def find_nearest_levels(prices, price, window=SWING_WINDOW):
    """أقرب مقاومة فوق السعر وأقرب دعم تحته، من تاريخ الأسعار المخزّن"""
    arr = list(prices)
    n = len(arr)
    highs, lows = [], []
    for j in range(window, n - window):
        segment = arr[j - window:j + window + 1]
        if arr[j] == max(segment):
            highs.append(arr[j])
        if arr[j] == min(segment):
            lows.append(arr[j])
    resistance = min([h for h in highs if h > price], default=None)
    support = max([low for low in lows if low < price], default=None)
    return support, resistance


def default_stock_state(cash_slot):
    return {
        "prices": [],
        "cash": cash_slot,
        "shares": 0,
        "avg_buy_price": None,
        "peak_price": None,  # لتتبع أعلى قمة من بعد الشرا (لـ trailing stop)
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
            stocks[ticker].setdefault("peak_price", None)

    return {"stocks": stocks}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# سجل الصفقات التاريخي (trade_log.csv) - بيتراكم بمرور الوقت
# ============================================================
def append_trade_log(ticker, action, price, shares, amount, profit, confidence):
    file_exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "ticker", "action", "price", "shares",
                "amount", "profit", "confidence_after",
            ])
        writer.writerow([
            datetime.now(ZoneInfo("Africa/Cairo")).strftime("%Y-%m-%d %H:%M:%S"),
            ticker, action, f"{price:.2f}", shares,
            f"{amount:.2f}", f"{profit:.2f}" if profit is not None else "",
            f"{confidence:.2f}",
        ])


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


def fetch_price(ticker, prev_price=None):
    """
    بيجيب السعر من Mubasher، ولو فشل بيرجع لـ yfinance. كمان بيعمل
    فحص أمان: لو السعر الجديد مختلف بشكل غريب عن آخر سعر معروف،
    بيتأكد من yfinance قبل ما يصدقه، عشان نتجنب سعر غلط بسبب تغيير
    في شكل صفحة Mubasher (الـ regex بيدور على رقم بصيغة معينة، ولو
    الصفحة اتغيرت ممكن ياخد رقم غلط زي أعلى/أقل سعر باليوم بدل السعر
    الفعلي).
    """
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
                print("  [تنبيه] yfinance بيأكد إن فيه حركة حقيقية، هكمل بسعر Mubasher")
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


def fallback_explanation(action, price, sma, zscore):
    if action == "BUY":
        return f"السعر ({price:.2f}) فوق المتوسط ({sma:.2f})، إشارة اتجاه صاعد."
    if action == "SELL_SIGNAL":
        return "الإشارة رجعت محايدة أو هابطة، فتم البيع."
    if action == "SELL_STOP_LOSS":
        return "تم البيع تلقائياً بسبب وصول الخسارة لحد وقف الخسارة."
    if action == "SELL_TAKE_PROFIT":
        return "تم البيع تلقائياً بعد الوصول لهدف الربح المحدد."
    if action == "SELL_TRAILING_STOP":
        return "تم البيع تلقائياً لأن السعر رجع للخلف بنسبة كافية من أعلى قمة وصلها بعد الشرا."
    if action == "SELL_TREND_BREAK":
        return "تم البيع لأن السعر كسر الترند طويل المدى (تحت المتوسط الطويل)، مش مجرد تذبذب قصير."
    if action == "SELL_SUPPORT_BREAK":
        return "تم البيع لأن السعر كسر نقطة دعم معروفة، إشارة ضعف في الترند."
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
    prev_price = stock_state["prices"][-1] if stock_state["prices"] else None
    try:
        price = fetch_price(ticker, prev_price=prev_price)
    except Exception as e:
        print(f"  خطأ في جلب السعر: {e}")
        return None

    value = stock_state["cash"] + stock_state["shares"] * price

    # النافذة بتاعة الـ SMA/الانحراف المعياري بتاخد الأسعار اللي *قبل*
    # السعر الحالي بس (زي الباكتست بالظبط) - عشان السعر الحالي ميأثرش
    # في المتوسط اللي بيتقارن بيه هو نفسه (lookahead bias). بنخزن
    # تاريخ أطول (HISTORY_WINDOW) عشان لو STRATEGY_MODE=trend_following
    # يحتاج فلتر 50 يوم يبقى متاح
    history = stock_state["prices"]
    if len(history) < LOOKBACK_WINDOW:
        history.append(price)
        stock_state["prices"] = history[-HISTORY_WINDOW:]
        print(f"  بنجمع بيانات: {len(stock_state['prices'])}/{LOOKBACK_WINDOW}  |  "
              f"السعر: {price:.2f}  |  قيمة الخزنة: {value:,.2f} جنيه")
        return None

    arr = np.array(history[-LOOKBACK_WINDOW:])
    sma = arr.mean()
    std = arr.std()

    sma_long = None
    if STRATEGY_MODE in ("trend_following", "trend_adx_sr") and len(history) >= TREND_FILTER_WINDOW:
        sma_long = np.array(history[-TREND_FILTER_WINDOW:]).mean()

    # وقف خسارة وهدف ربح مبنيين على تقلب السهم نفسه (مش نسبة ثابتة)
    volatility_ratio = (std / sma) if sma > 0 else 0.02
    stop_loss_pct = min(max(2.5 * volatility_ratio, 0.02), 0.08)
    take_profit_pct = stop_loss_pct * 2  # نسبة مخاطرة:عائد 1:2

    momentum_signal = 1 if price > sma else -1
    zscore = (price - sma) / std if std > 0 else 0

    if STRATEGY_MODE in ("trend_following", "trend_adx_sr"):
        if sma_long is None:
            signal = 0  # لسه معندناش بيانات كفاية لفلتر الـ 50 يوم
        elif price > sma and price > sma_long:
            signal = 1
            if STRATEGY_MODE == "trend_adx_sr":
                adx_value = compute_adx_lite(history)
                if adx_value is None or adx_value < ADX_THRESHOLD:
                    signal = 0  # الترند مش مؤكد بقوة كفاية - تجنب سوق متذبذب
                else:
                    support, resistance = find_nearest_levels(history, price)
                    if resistance is not None and price < resistance:
                        if (resistance - price) / price < SR_BUFFER_PCT:
                            signal = 0  # قريب جدًا من مقاومة - خطر ارتداد
        elif price < sma_long:
            signal = -1
        else:
            signal = 0  # تذبذب قصير المدى داخل ترند أكبر - محايد عمدًا
    else:
        if STRATEGY_MODE in ("momentum_only", "momentum_trailing"):
            raw_score = momentum_signal
        else:
            if zscore > MEAN_REVERSION_STD:
                mr_signal = -1
            elif zscore < -MEAN_REVERSION_STD:
                mr_signal = 1
            else:
                mr_signal = 0
            raw_score = 0.6 * momentum_signal + 0.4 * mr_signal

        if raw_score > SIGNAL_THRESHOLD:
            signal = 1
        elif raw_score < -SIGNAL_THRESHOLD:
            signal = -1
        else:
            signal = 0

    action = None
    extra = ""

    # وقف خسارة / هدف ربح - لهم الأولوية
    if stock_state["shares"] > 0 and stock_state["avg_buy_price"]:
        if STRATEGY_MODE in ("momentum_trailing", "trend_following", "trend_adx_sr"):
            # وقف خسارة متحرّك: بيتبع أعلى سعر وصله السهم من بعد
            # الشرا (مش سعر الشرا نفسه)، عشان يسيب الربح يكبر في
            # الترندات الطويلة بدل ما يقفل المركز عند هدف ثابت
            stock_state["peak_price"] = max(stock_state["peak_price"] or price, price)
            change_from_peak = (price - stock_state["peak_price"]) / stock_state["peak_price"]
            if change_from_peak < -stop_loss_pct:
                action = "SELL_TRAILING_STOP"
        else:
            change_pct = (price - stock_state["avg_buy_price"]) / stock_state["avg_buy_price"]
            if change_pct < -stop_loss_pct:
                action = "SELL_STOP_LOSS"
            elif change_pct > take_profit_pct:
                action = "SELL_TAKE_PROFIT"

        # خروج لو سعر الدعم المعروف اتكسر فعليًا (trend_adx_sr بس)
        if action is None and STRATEGY_MODE == "trend_adx_sr":
            support, _ = find_nearest_levels(history, price)
            if support is not None and price < support:
                action = "SELL_SUPPORT_BREAK"

        # خروج بسبب الإشارة نفسها (منفصل عن وقف الخسارة/الهدف)
        if action is None:
            if STRATEGY_MODE in ("trend_following", "trend_adx_sr"):
                # في trend_following/trend_adx_sr، بس انكسار الترند
                # الكبير (signal == -1) بيقفل المركز - التذبذب البسيط
                # (signal == 0) بيتجاهله عمدًا
                if signal == -1:
                    action = "SELL_TREND_BREAK"
            else:
                if signal <= 0:
                    action = "SELL_SIGNAL"

        if action:
            gross_proceeds = stock_state["shares"] * price
            fee = gross_proceeds * TRANSACTION_COST_PCT
            proceeds = gross_proceeds - fee
            profit = proceeds - (stock_state["shares"] * stock_state["avg_buy_price"])
            profit_pct = profit / (stock_state["shares"] * stock_state["avg_buy_price"])
            sold_shares = stock_state["shares"]
            stock_state["cash"] += proceeds
            stock_state["shares"] = 0
            stock_state["avg_buy_price"] = None
            stock_state["peak_price"] = None

            # تحديث درجة الثقة بناءً على النتيجة
            if profit >= 0:
                stock_state["confidence"] = min(MAX_CONFIDENCE, stock_state["confidence"] * 1.1)
            else:
                stock_state["confidence"] = max(MIN_CONFIDENCE, stock_state["confidence"] * 0.85)

            label = {
                "SELL_TAKE_PROFIT": "هدف الربح",
                "SELL_STOP_LOSS": "وقف الخسارة",
                "SELL_TRAILING_STOP": "وقف الخسارة المتحرّك",
                "SELL_TREND_BREAK": "انكسار الترند طويل المدى",
                "SELL_SUPPORT_BREAK": "كسر نقطة الدعم",
                "SELL_SIGNAL": "إشارة اتجاه",
            }.get(action, action)
            extra = (
                f"باع {sold_shares} سهم بسعر {price:.2f} جنيه (وصل {label})\n"
                f"إجمالي البيع: {gross_proceeds:,.2f} جنيه (عمولة: {fee:,.2f} جنيه)\n"
                f"{'ربح' if profit >= 0 else 'خسارة'}: {profit:,.2f} جنيه ({profit_pct:+.2%})\n"
                f"كاش الخزنة دلوقتي: {stock_state['cash']:,.2f} جنيه"
            )
            append_trade_log(ticker, action, price, sold_shares, proceeds, profit, stock_state["confidence"])

    # شراء
    if action is None and stock_state["shares"] == 0 and signal == 1:
        usable_cash = stock_state["cash"] * stock_state["confidence"]
        # نحسب عدد الأسهم مع الأخذ في الاعتبار عمولة الشرا، عشان
        # التكلفة الكاملة (سهم + عمولة) متتجاوزش الكاش المتاح
        shares_to_buy = int(usable_cash // (price * (1 + TRANSACTION_COST_PCT)))
        if shares_to_buy > 0:
            gross_cost = shares_to_buy * price
            fee = gross_cost * TRANSACTION_COST_PCT
            total_cost = gross_cost + fee
            stock_state["cash"] -= total_cost
            stock_state["shares"] = shares_to_buy
            # متوسط سعر الشرا بيشمل العمولة، عشان حساب الربح لاحقاً
            # يبقى صافي فعلاً من غير ما نحتاج نتتبع العمولة لوحدها
            stock_state["avg_buy_price"] = total_cost / shares_to_buy
            stock_state["peak_price"] = price
            action = "BUY"
            extra = (
                f"اشترى {shares_to_buy} سهم بسعر {price:.2f} جنيه "
                f"(إجمالي {gross_cost:,.2f} جنيه + عمولة {fee:,.2f} جنيه)\n"
                f"درجة الثقة الحالية: {stock_state['confidence']:.0%}\n"
                f"باقي كاش الخزنة: {stock_state['cash']:,.2f} جنيه"
            )
            append_trade_log(ticker, action, price, shares_to_buy, total_cost, None, stock_state["confidence"])

    value = stock_state["cash"] + stock_state["shares"] * price
    holding = f"{stock_state['shares']} سهم" if stock_state["shares"] > 0 else "كاش بالكامل"
    print(f"  السعر: {price:.2f} | SMA: {sma:.2f} | Z: {zscore:+.2f} | "
          f"وقف خسارة: {stop_loss_pct:.1%} | هدف ربح: {take_profit_pct:.1%} | "
          f"المحفظة: {holding} | القيمة: {value:,.2f} جنيه | ثقة: {stock_state['confidence']:.0%}")

    # نسجل السعر الحالي في النافذة التاريخية *بعد* اتخاذ القرار،
    # عشان يبقى متاح للمرة الجاية من غير ما يأثر في قرار النهاردة
    stock_state["prices"].append(price)
    stock_state["prices"] = stock_state["prices"][-HISTORY_WINDOW:]

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
