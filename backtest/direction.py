"""
تصنيف شراء/بيع لكل سهم: هل الموديل شايف إن السهم هيكسب (شراء) ولا هيخسر (بيع) في الفترة دي؟

ترتيب XGBoost الأصلي نسبي بين الأسهم (مين أحسن من مين)، ومبيقولش لوحده إن السهم هيطلع ولا ينزل.
فهنا موديل تاني (XGBoost classifier) بيتدرب على سؤال واحد: "العائد المطلق للفترة ده هيبقى موجب؟"،
بنفس المدخلات (ومنها مدخلات حالة السوق كله، فبيعرف لو السوق نازل).
  احتمال صعود >= 50%  ->  شراء
  احتمال صعود <  50%  ->  بيع
والاحتمال نفسه مش بيتعرض (قرار سابق: من غير نسب احتمال في التقارير).

holdout(): بيدرّب نفس الموديل على الماضي بس، وبيجربه على آخر `test_days` يوم ماشافهمش (مع embargo)،
عشان الرسالة تقول بصراحة دقة التصنيف ده فعلاً، ومقارنتها بالتخمين الثابت.
"""

import numpy as np

BUY, SELL = "شراء", "بيع"


def _model(trees):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=trees, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
        min_child_weight=50, reg_lambda=10.0, tree_method="hist", n_jobs=2, random_state=0,
        eval_metric="logloss")


def fit_predict(Xcs, ret, E, t, cols, trees=150):
    """بيدرّب على كل الصفوف الصالحة (E) وبيرجّع احتمال الصعود لأسهم `cols` في اليوم t."""
    y = (ret[E] > 0).astype(int)
    if len(y) < 5000 or y.min() == y.max():
        return None
    m = _model(trees)
    m.fit(Xcs[E], y)
    return m.predict_proba(Xcs[t, cols])[:, 1]


def label(p):
    return BUY if p >= 0.5 else SELL


def holdout(Xcs, ret, E, h, trees=150, test_days=250):
    """دقة التصنيف على آخر `test_days` يوم (التدريب قبلهم بـ h يوم embargo). بيرجّع dict أو None."""
    T = Xcs.shape[0]
    start = T - test_days
    cut = start - h
    if cut < 300:
        return None
    Etr = E.copy()
    Etr[cut:] = False
    ytr = (ret[Etr] > 0).astype(int)
    if len(ytr) < 5000 or ytr.min() == ytr.max():
        return None
    Ete = np.zeros_like(E)
    Ete[start:] = E[start:]
    if int(Ete.sum()) < 500:
        return None
    m = _model(trees)
    m.fit(Xcs[Etr], ytr)
    p = m.predict_proba(Xcs[Ete])[:, 1]
    y = (ret[Ete] > 0).astype(int)
    pred = (p >= 0.5).astype(int)
    up = float(y.mean())
    out = {"days": int(test_days), "n": int(len(y)),
           "acc": float((pred == y).mean() * 100),
           "base": float(max(up, 1 - up) * 100),         # أحسن تخمين ثابت (دايمًا الأغلب)، بأثر رجعي
           "buy_share": float(pred.mean() * 100)}
    out["buy_hit"] = float(y[pred == 1].mean() * 100) if pred.sum() else float("nan")
    out["sell_hit"] = float((1 - y[pred == 0]).mean() * 100) if (pred == 0).sum() else float("nan")
    return out


def holdout_line(ho):
    """سطر الرسالة اللي بيقول دقة التصنيف بصراحة."""
    if not ho:
        return "دقة تصنيف شراء/بيع: مش قادر أقيسها النهاردة (بيانات قليلة)."
    verdict = ("أحسن من التخمين الثابت" if ho["acc"] >= ho["base"] + 1.0
               else "مش أحسن من التخمين الثابت")
    return (f"دقة شراء/بيع في آخر {ho['days']} يوم (على أيام الموديل ماشافهاش): {ho['acc']:.0f}% صح "
            f"مقابل {ho['base']:.0f}% لو خمّنا دايمًا الأغلب، يعني {verdict}. "
            f"لما قال شراء: طلع {ho['buy_hit']:.0f}% من المرات | لما قال بيع: نزل {ho['sell_hit']:.0f}%")
