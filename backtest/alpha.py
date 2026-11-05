"""مدخلات إضافية على طريقة Alpha158 (معيار Qlib): شكل الشمعة، متوسطات ونطاقات، RSV، aroon، ارتباط السعر بالحجم، عدّ أيام الصعود، RSI-like، وحجم التداول.
كلها مبنية من O/H/L/C/V وبعدين بتترتّب بين الأسهم (cross-sectional) زي باقي المدخلات."""
import numpy as np
import pandas as pd


def _aroon(df, w, kind):
    f = (lambda a: float(np.argmax(a))) if kind == "max" else (lambda a: float(np.argmin(a)))
    return df.rolling(w, min_periods=w).apply(f, raw=True) / (w - 1)


def build_alpha(C, H, L, V, O, liquid, csrank):
    """بيرجّع (مصفوفة (T,N,K) مترتبة, أسماء)."""
    C, H, L, O = (x.where(x > 0) for x in (C, H, L, O))
    V = V.where(V >= 0)
    TO = (C * V)                                  # قيمة التداول
    F = {}
    F["kmid"] = (C - O) / O
    F["klen"] = (H - L) / O
    F["kup"] = (H - np.maximum(O, C)) / O
    F["klow"] = (np.minimum(O, C) - L) / O
    F["ksft"] = (2 * C - H - L) / O
    r1 = C.pct_change(fill_method=None).clip(-0.15, 0.15)
    dv = np.log1p(V)
    for w in (5, 10, 20, 60):
        F[f"ma{w}"] = C.rolling(w, min_periods=w).mean() / C
        F[f"max{w}"] = H.rolling(w, min_periods=w).max() / C
        F[f"min{w}"] = L.rolling(w, min_periods=w).min() / C
        hi, lo = H.rolling(w, min_periods=w).max(), L.rolling(w, min_periods=w).min()
        F[f"rsv{w}"] = (C - lo) / (hi - lo + 1e-9)
        F[f"cntp{w}"] = (r1 > 0).astype(float).where(r1.notna()).rolling(w, min_periods=w).mean()
        F[f"vma{w}"] = TO.rolling(w, min_periods=w).mean() / (TO + 1)
    for w in (5, 60):
        F[f"std{w}"] = r1.rolling(w, min_periods=w).std()
    for w in (10, 20, 60):
        F[f"corr{w}"] = C.rolling(w, min_periods=w).corr(dv)
        F[f"cord{w}"] = r1.rolling(w, min_periods=w).corr(dv.diff())
        F[f"vstd{w}"] = TO.rolling(w, min_periods=w).std() / (TO.rolling(w, min_periods=w).mean() + 1)
    for w in (20, 60):
        F[f"qtlu{w}"] = C.rolling(w, min_periods=w).quantile(0.8) / C
        F[f"qtld{w}"] = C.rolling(w, min_periods=w).quantile(0.2) / C
        F[f"imax{w}"] = _aroon(H, w, "max")
        F[f"imin{w}"] = _aroon(L, w, "min")
    d = C.diff()
    for w in (5, 20):
        up = d.clip(lower=0).rolling(w, min_periods=w).sum()
        tot = d.abs().rolling(w, min_periods=w).sum()
        F[f"sump{w}"] = up / (tot + 1e-9)
    names = list(F)
    X = np.stack([csrank(F[k].replace([np.inf, -np.inf], np.nan), liquid).values for k in names], axis=2).astype(np.float32)
    return X, names
