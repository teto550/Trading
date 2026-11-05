"""موديلات الترتيب المستخدمة في التقارير الحية (نفس إعدادات ensemble_test.py اللي اتقاست)."""
import numpy as np

# اللي أثبتته التجربة (2022+، تكلفة 20 bps): 20 يوم → LightGBM لوحده، 10 أيام → متوسط ترتيب 3 موديلات، 5 أيام → XGBoost
SPEC = {5: ("xgb",), 10: ("xgb", "lgbm", "cat"), 20: ("lgbm",), 40: ("lgbm",), 60: ("lgbm",)}


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


def fit_score(h, Xtr, ytr, w, Xnow, trees=150):
    """بيدرّب موديلات الفترة h ويرجّع (درجة الترتيب 0..1 لكل سهم النهاردة، أسماء الموديلات اللي اشتغلت)."""
    scores, used = [], []
    for kind in SPEC.get(h, ("xgb",)):
        try:
            m = make(kind, trees)
            m.fit(Xtr, ytr, sample_weight=w)
            p = m.predict(Xnow)
            scores.append(np.argsort(np.argsort(p)) / max(len(p) - 1, 1))
            used.append(kind)
        except Exception as e:  # noqa: BLE001
            print(f"[models] {kind} فشل لفترة {h}: {str(e)[:120]}")
    if not scores:                                   # احتياطي أخير: XGBoost
        m = make("xgb", trees)
        m.fit(Xtr, ytr, sample_weight=w)
        p = m.predict(Xnow)
        return np.argsort(np.argsort(p)) / max(len(p) - 1, 1), ["xgb"]
    return np.mean(scores, axis=0), used
