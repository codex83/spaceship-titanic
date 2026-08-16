import json
import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
import catboost as cb

from train_model import build_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_FOLDS = 3  # fewer folds during search for speed; final run uses 5


def cv_score_lgb(params, X, y, groups, cat_cols):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    for tr_idx, val_idx in sgkf.split(X, y, groups):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = lgb.LGBMClassifier(**params, random_state=SEED, verbosity=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_error",
                   categorical_feature=cat_cols, callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = model.predict_proba(X_val)[:, 1]
        accs.append(((pred > 0.5).astype(int) == y_val.values).mean())
    return float(np.mean(accs))


def cv_score_cat(params, X, y, groups, cat_cols):
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    for tr_idx, val_idx in sgkf.split(X, y, groups):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = cb.CatBoostClassifier(**params, random_seed=SEED, verbose=False,
                                        eval_metric="Accuracy", early_stopping_rounds=100)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_idx, use_best_model=True)
        pred = model.predict_proba(X_val)[:, 1]
        accs.append(((pred > 0.5).astype(int) == y_val.values).mean())
    return float(np.mean(accs))


def tune_lgb(X, y, groups, cat_cols, n_trials=40):
    def objective(trial):
        params = dict(
            n_estimators=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        )
        return cv_score_lgb(params, X, y, groups, cat_cols)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"LGB best acc={study.best_value:.4f}")
    return study.best_params


def tune_cat(X, y, groups, cat_cols, n_trials=40):
    def objective(trial):
        params = dict(
            iterations=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0, 1),
            random_strength=trial.suggest_float("random_strength", 0, 2),
        )
        return cv_score_cat(params, X, y, groups, cat_cols)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"CAT best acc={study.best_value:.4f}")
    return study.best_params


def main():
    X, y, X_test, groups, cat_cols, test_ids = build_features()

    print("Tuning LightGBM...")
    lgb_params = tune_lgb(X, y, groups, cat_cols)
    print("Best LGB params:", lgb_params)

    print("\nTuning CatBoost...")
    cat_params = tune_cat(X, y, groups, cat_cols)
    print("Best CAT params:", cat_params)

    with open("best_params.json", "w") as f:
        json.dump({"lgb": lgb_params, "cat": cat_params}, f, indent=2)
    print("\nSaved best_params.json")


if __name__ == "__main__":
    main()
