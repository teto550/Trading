"""
مدخلات الاقتصاد الكلي للموديل: مؤشر EGX30 وسعر الدولار مقابل الجنيه (من Yahoo).

مدخلات على مستوى السوق (نفس القيمة لكل الأسهم في اليوم):
  egx_r5 / egx_r20 : تغير EGX30 في آخر 5 و20 يوم
  egx_vs50         : بُعد EGX30 عن متوسطه 50 يوم (اتجاه السوق)
  usd_r5 / usd_r20 : تغير الدولار مقابل الجنيه في آخر 5 و20 يوم
ومدخل لكل سهم:
  beta60           : حساسية السهم لحركة EGX30 في آخر 60 يوم (مرتب بين الأسهم)
فايدة beta60 إنه بيخلّي الموديل يفرّق بين سهم بيتحرك مع السوق وسهم دفاعي لما السوق يتغير.

كل المدخلات بتستخدم بيانات لحد اليوم t بس. لو Yahoo رفض سلسلة (أو الاتنين)، الموديل بيكمّل بالباقي
ومن غيرهم لو مفيش، ولازم الكود اللي بيستدعي يقول ده في الرسالة.
"""

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MACRO_TICKERS = {"egx30": ["^CASE30", "EGX30.CA", "^EGX30"], "usd": ["EGP=X"]}
MACRO_MARKET = ["egx_r5", "egx_r20", "egx_vs50", "usd_r5", "usd_r20"]


def fetch_macro(fresh=False):
    """بيرجّع DataFrame (index تاريخ، أعمدة egx30 و/أو usd). الأعمدة اللي فشل جلبها بتتشال."""
    import yfinance as yf
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "macro.pkl")
    if not fresh and os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 86400:
        return pd.read_pickle(cache)
    cols = {}
    for key, tks in MACRO_TICKERS.items():
        for tk in tks:
            for attempt in range(2):
                try:
                    h = yf.Ticker(tk).history(period="max", interval="1d", auto_adjust=True)
                    if len(h) > 200:
                        idx = pd.to_datetime(h.index).tz_localize(None).normalize()
                        s = h["Close"].copy()
                        s.index = idx
                        s = s[~s.index.duplicated(keep="last")]
                        cols[key] = s.where(s > 0)
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(2)
            print(f"[macro] {key} ({tk}): {'تمام ' + str(len(cols[key])) + ' يوم' if key in cols else 'فشل'}")
            if key in cols:
                break
    M = pd.DataFrame(cols).sort_index()
    if len(M.columns):
        pd.to_pickle(M, cache)
    return M


def proxy_index(C, V, n=30):
    """مؤشر تقريبي لـEGX30 لو Yahoo رفض المؤشر الحقيقي: عائد أكبر n سهم سيولة (متوسط تداول 20 يوم،
    بوزن السيولة، والترتيب من امبارح فمفيش تسريب). مش نفس المؤشر الرسمي بس بيتحرك زيه تقريبًا."""
    r = C / C.shift(1) - 1
    r = r.where(r.abs() <= 0.25)
    tov = (C * V).rolling(20, min_periods=15).median().shift(1)
    rank = tov.rank(axis=1, ascending=False)
    w = tov.where(rank <= n)
    rt = (r * w).sum(axis=1, min_count=1) / w.where(r.notna()).sum(axis=1)
    return 1000 * (1 + rt.fillna(0)).cumprod()


def ensure_egx30(M, C, V):
    """بيضمن وجود عمود egx30: الحقيقي من Yahoo لو اتجاب، وإلا التقريبي. بيرجّع (M, المصدر)."""
    M = pd.DataFrame(index=C.index) if M is None else M
    if "egx30" in M.columns:
        return M, "yahoo"
    M = M.reindex(M.index.union(C.index))      # الدولار بيتداول أيام غير أيام البورصة: لازم الاتنين في نفس الفهرس
    M["egx30"] = proxy_index(C, V).reindex(M.index)
    return M, "proxy"


def _align(M, index):
    u = M.index.union(index)
    return M.reindex(u).sort_index().ffill(limit=5).reindex(index)


def build_X(f, mkt, liquid, N, xf, mf, M, csrank):
    """بيبني مصفوفة المدخلات (T, N, K). بيرجّع (Xcs, أسماء المدخلات).
    لو M فاضي أو None بيرجّع نفس المدخلات الأساسية بالظبط."""
    cs = {k: csrank(f[k], liquid).values for k in xf}
    stock = [cs[k] for k in xf]
    s_names = list(xf)
    market = [np.repeat(mkt[k].values[:, None], N, axis=1) for k in mf]
    m_names = list(mf)
    if M is not None and len(M.columns):
        A = _align(M, liquid.index)
        extra = {}
        if "egx30" in A:
            e = A["egx30"]
            le = np.log(e)
            extra["egx_r5"] = le.diff(5)
            extra["egx_r20"] = le.diff(20)
            extra["egx_vs50"] = e / e.rolling(50, min_periods=40).mean() - 1
            re = e.pct_change(fill_method=None).clip(-0.15, 0.15)
            beta = f["r1"].rolling(60, min_periods=40).cov(re).div(re.rolling(60, min_periods=40).var(), axis=0)
            stock.append(csrank(beta, liquid).values)
            s_names.append("beta60")
        if "usd" in A:
            lu = np.log(A["usd"])
            extra["usd_r5"] = lu.diff(5)
            extra["usd_r20"] = lu.diff(20)
        for k in MACRO_MARKET:
            if k in extra:
                market.append(np.repeat(extra[k].values[:, None], N, axis=1))
                m_names.append(k)
    Xcs = np.stack(stock + market, axis=2).astype(np.float32)
    return Xcs, s_names + m_names


def synthetic_macro(mkt1, seed=7):
    """سلسلة EGX30 ودولار اصطناعية لاختبار السلامة بس (EGX30 = تراكم متوسط عوائد الأسهم)."""
    rng = np.random.default_rng(seed)
    egx = 1000 * (1 + mkt1.fillna(0)).cumprod()
    usd = 15 * np.exp(np.cumsum(rng.normal(0.0002, 0.003, len(mkt1))))
    return pd.DataFrame({"egx30": egx, "usd": pd.Series(usd, index=mkt1.index)})
