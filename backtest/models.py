"""موديلات الترتيب المستخدمة في التقارير الحية (نفس إعدادات ensemble_test.py اللي اتقاست)."""
import numpy as np

# اللي أثبتته التجربة (2022+، تكلفة 20 bps): 20 يوم → LightGBM لوحده، 10 أيام → متوسط ترتيب 3 موديلات، 5 أيام → XGBoost
SPEC = {1: ("xgb", "lgbm", "cat"), 5: ("xgb",), 10: ("xgb", "lgbm", "cat"), 20: ("lgbm",), 40: ("lgbm",), 60: ("lgbm",)}


def make(kind, trees=150, seed=0):
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=trees, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
                                min_child_weight=50, reg_lambda=10.0, tree_method="hist", n_jobs=2, random_state=seed)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=trees, num_leaves=15, learning_rate=0.05, subsample=0.7, subsample_freq=1,
                                 colsample_bytree=0.7, min_child_samples=100, reg_lambda=10.0, n_jobs=2,
                                 random_state=seed, verbose=-1)
    import catboost as cb
    return cb.CatBoostRegressor(iterations=trees, depth=4, learning_rate=0.06, l2_leaf_reg=10, thread_count=2,
                                random_seed=seed, verbose=0, allow_writing_files=False)


ALL = ("xgb", "lgbm", "cat")
NAMES = {"xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost", "ens": "متوسط الـ3 (ensemble)"}


def fit_scores(h, Xtr, ytr, w, Xnow, trees=150):
    """بيدرّب كل موديل لوحده ويرجّع {موديل: درجة ترتيب 0..1 لكل سهم النهاردة} + "ens" (متوسطهم).
    بنحفظ ترشيحات كل موديل عشان نعرف بعد كده مين كان أحسن فعلاً."""
    sc = {}
    for kind in ALL:
        try:
            m = make(kind, trees)
            m.fit(Xtr, ytr, sample_weight=w)
            p = m.predict(Xnow)
            sc[kind] = np.argsort(np.argsort(p)) / max(len(p) - 1, 1)
        except Exception as e:  # noqa: BLE001
            print(f"[models] {kind} فشل لفترة {h}: {str(e)[:120]}")
    if sc:
        sc["ens"] = np.mean([sc[k] for k in ALL if k in sc], axis=0)
    return sc


def choose(h, sc):
    """الموديل (أو الموديلات) المعتمد للعرض لفترة h، حسب نتايج الاختبار. بيرجّع (درجة, أسماء)."""
    kinds = [k for k in SPEC.get(h, ("xgb",)) if k in sc]
    if not kinds:
        kinds = [k for k in ALL if k in sc][:1]
    if not kinds:
        raise RuntimeError("كل الموديلات فشلت")
    return np.mean([sc[k] for k in kinds], axis=0), kinds


def top_by_model(sc, cols, names, top=10):
    """أسماء أعلى `top` أسهم عند كل موديل (للمقارنة الحية بعدين)."""
    out = {}
    for k, v in sc.items():
        order = np.argsort(-v)[:top]
        out[k] = [str(names[int(cols[i])]).replace(".CA", "") for i in order]
    return out


def perf(by_model, get_ret):
    """أداء قايمة كل موديل: {موديل: {"n","up","avg"}}. get_ret(name) → العائد % أو None."""
    out = {}
    for k, nm in (by_model or {}).items():
        rs = [r for r in (get_ret(n) for n in nm) if r is not None]
        if len(rs) >= 3:
            out[k] = {"n": len(rs), "up": sum(1 for r in rs if r > 0), "avg": round(float(np.mean(rs)), 3)}
    return out


def lines(rec, history):
    """سطور رسالة النتيجة: ترتيب الموديلات في المرة دي + الترتيب التراكمي (فرق متوسط عائد أعلى 10 عن السوق)."""
    ms = rec.get("models") or {}
    if len(ms) < 2:
        return []
    L_ = ["", "🏁 أي موديل كان أحسن في المرة دي (أعلى 10 عند كل موديل):"]
    for k, v in sorted(ms.items(), key=lambda kv: -kv[1]["avg"]):
        L_.append(f"• {NAMES.get(k, k)}: متوسط {v['avg']:+.2f}% ({v['avg'] - rec['mkt_avg']:+.2f}% عن السوق) | طلع {v['up']} من {v['n']}")
    acc = {}
    for d in history + [rec]:
        for k, v in (d.get("models") or {}).items():
            e = acc.setdefault(k, [])
            e.append(v["avg"] - d["mkt_avg"])
    n = max((len(v) for v in acc.values()), default=0)
    if n >= 2:
        L_.append(f"الترتيب التراكمي ({n} نتيجة): " + " | ".join(
            f"{NAMES.get(k, k)} {sum(v) / len(v):+.2f}% (أحسن من السوق {sum(1 for x in v if x > 0)}/{len(v)})"
            for k, v in sorted(acc.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))))
    return L_


def fit_score(h, Xtr, ytr, w, Xnow, trees=150):
    """نفس الواجهة القديمة: (درجة الموديل المعتمد، أسماؤه)."""
    sc = fit_scores(h, Xtr, ytr, w, Xnow, trees)
    if not sc:                                       # احتياطي أخير
        m = make("xgb", trees)
        m.fit(Xtr, ytr, sample_weight=w)
        p = m.predict(Xnow)
        return np.argsort(np.argsort(p)) / max(len(p) - 1, 1), ["xgb"]
    return choose(h, sc)
