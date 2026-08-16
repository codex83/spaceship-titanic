"""Isolated ablation tests for 5 candidate improvements, each measured
against the current baseline feature set with fixed tuned CatBoost params."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import catboost as cb

from train_model import build_features, TUNED, SEED, N_FOLDS

CAT_PARAMS = dict(iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3)
CAT_PARAMS.update(TUNED.get("cat", {}))


def cv_acc(X, y, groups, cat_cols, seed=SEED, n_folds=N_FOLDS):
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for tr_idx, val_idx in sgkf.split(X, y, groups):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = cb.CatBoostClassifier(**CAT_PARAMS, random_seed=seed, verbose=False,
                                        eval_metric="Accuracy", early_stopping_rounds=150)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_idx, use_best_model=True)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
    return oof


def acc(oof, y, thresh=0.5):
    return ((oof > thresh).astype(int) == y.values).mean()


def main():
    X, y, X_test, groups, cat_cols, test_ids = build_features()

    print("=== Baseline ===")
    base_oof = cv_acc(X, y, groups, cat_cols)
    base_acc = acc(base_oof, y)
    print(f"Baseline OOF acc: {base_acc:.4f}\n")

    results = {"baseline": base_acc}

    # --- Idea 1: out-of-fold target encoding for categorical columns ---
    print("=== Idea 1: OOF target encoding ===")
    X1 = X.copy()
    te_cols = ["HomePlanet", "Deck", "Side", "Destination", "CabinRegion", "AgeGroup"]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    global_mean = y.mean()
    for c in te_cols:
        X1[f"{c}_te"] = np.nan
    for tr_idx, val_idx in sgkf.split(X, y, groups):
        tr_y = y.iloc[tr_idx]
        for c in te_cols:
            means = tr_y.groupby(X[c].iloc[tr_idx].values).mean()
            counts = tr_y.groupby(X[c].iloc[tr_idx].values).count()
            smooth = (means * counts + global_mean * 20) / (counts + 20)
            X1.loc[X1.index[val_idx], f"{c}_te"] = X[c].iloc[val_idx].map(smooth).fillna(global_mean).values
    oof1 = cv_acc(X1, y, groups, cat_cols)
    a1 = acc(oof1, y)
    print(f"Idea 1 OOF acc: {a1:.4f} (delta {a1 - base_acc:+.4f})\n")
    results["target_encoding"] = a1

    # --- Idea 2: log1p transform of skewed spend features ---
    print("=== Idea 2: log1p spend transforms ===")
    X2 = X.copy()
    for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
              "TotalSpend", "LuxurySpend", "GroupTotalSpend", "GroupAvgSpend"]:
        X2[f"{c}_log"] = np.log1p(X2[c])
    oof2 = cv_acc(X2, y, groups, cat_cols)
    a2 = acc(oof2, y)
    print(f"Idea 2 OOF acc: {a2:.4f} (delta {a2 - base_acc:+.4f})\n")
    results["log_transform"] = a2

    # --- Idea 3: consistency/interaction features ---
    print("=== Idea 3: consistency + interaction features ===")
    X3 = X.copy()
    X3["CryoSpendInconsistent"] = ((X3["CryoSleep"] == 1) & (X3["TotalSpend"] > 0)).astype(int)
    X3["CabinNumRankInDeck"] = X3.groupby("Deck")["CabinNum"].rank(pct=True)
    X3["HomePlanetDeck"] = X3["HomePlanet"].astype(str) + "_" + X3["Deck"].astype(str)
    cat_cols3 = cat_cols + ["HomePlanetDeck"]
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X3["HomePlanetDeck"] = le.fit_transform(X3["HomePlanetDeck"].astype(str))
    oof3 = cv_acc(X3, y, groups, cat_cols3)
    a3 = acc(oof3, y)
    print(f"Idea 3 OOF acc: {a3:.4f} (delta {a3 - base_acc:+.4f})\n")
    results["interactions"] = a3

    # --- Idea 4: multi-seed bagging (variance reduction check) ---
    print("=== Idea 4: multi-seed CV bagging ===")
    seed_accs = []
    for s in [42, 7, 123]:
        oof_s = cv_acc(X, y, groups, cat_cols, seed=s)
        a_s = acc(oof_s, y)
        seed_accs.append(a_s)
        print(f"  seed={s}: acc={a_s:.4f}")
    print(f"Idea 4 mean OOF acc across seeds: {np.mean(seed_accs):.4f} (std {np.std(seed_accs):.4f})\n")
    results["multi_seed_mean"] = float(np.mean(seed_accs))
    results["multi_seed_std"] = float(np.std(seed_accs))

    # --- Idea 5: threshold tuning ---
    print("=== Idea 5: decision threshold tuning ===")
    best_t, best_a = 0.5, base_acc
    for t in np.arange(0.30, 0.71, 0.01):
        a_t = acc(base_oof, y, thresh=t)
        if a_t > best_a:
            best_a, best_t = a_t, t
    print(f"Best threshold: {best_t:.2f}, acc={best_a:.4f} (delta {best_a - base_acc:+.4f})\n")
    results["threshold_tuning"] = best_a
    results["best_threshold"] = float(best_t)

    print("=== Summary ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
