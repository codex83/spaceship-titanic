import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
N_FOLDS = 5
SEED = 42


def engineer(df):
    """Runs on the full combined train+test frame so group/family aggregates
    see every passenger, including cross-file family links."""
    df = df.copy()

    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["Group"].transform("count")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]
    df["CabinRegion"] = (df["CabinNum"] // 300).fillna(-1)

    df["Surname"] = df["Name"].str.split().str[-1]
    df["IsNameMissing"] = df["Name"].isna().astype(int)
    # Family = same surname, possibly spanning multiple travel groups.
    df["FamilySize"] = df.groupby("Surname")["Surname"].transform("count")
    df["FamilySize"] = df["FamilySize"].where(df["Surname"].notna(), 1)
    df["FamilyGroupSpan"] = df.groupby("Surname")["Group"].transform("nunique")
    df["FamilyGroupSpan"] = df["FamilyGroupSpan"].where(df["Surname"].notna(), 1)

    for c in SPEND_COLS:
        df[c] = df[c].fillna(0)
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    for c in SPEND_COLS:
        df[f"{c}_ratio"] = np.where(df["TotalSpend"] > 0, df[c] / df["TotalSpend"], 0)
    df["LuxurySpend"] = df["Spa"] + df["VRDeck"]
    df["LuxuryRatio"] = np.where(df["TotalSpend"] > 0, df["LuxurySpend"] / df["TotalSpend"], 0)

    # CryoSleep passengers spend 0; use that to fill missing CryoSleep
    df.loc[df["CryoSleep"].isna() & (df["TotalSpend"] == 0), "CryoSleep"] = True
    df.loc[df["CryoSleep"].isna() & (df["TotalSpend"] > 0), "CryoSleep"] = False

    # Group-based imputation: fill HomePlanet/Destination from other members
    # of the same group, then fall back to same-surname family (families
    # usually share these even when split across groups).
    for c in ["HomePlanet", "Destination"]:
        grp_mode = df.groupby("Group")[c].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        df[c] = df[c].fillna(grp_mode)
        fam_mode = df.groupby("Surname")[c].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        df[c] = df[c].fillna(fam_mode)

    df["Age"] = df["Age"].fillna(df.groupby("HomePlanet")["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["AgeGroup"] = pd.cut(df["Age"], bins=[-1, 12, 18, 30, 50, 100],
                             labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"])

    for c in ["HomePlanet", "Destination", "Deck", "Side"]:
        df[c] = df[c].fillna("Unknown")

    df["VIP"] = df["VIP"].fillna(False)
    df["CryoSleep"] = df["CryoSleep"].fillna(False)
    df["CabinNum"] = df["CabinNum"].fillna(df["CabinNum"].median())

    # Group-level spend aggregates
    df["GroupTotalSpend"] = df.groupby("Group")["TotalSpend"].transform("sum")
    df["GroupAvgSpend"] = df.groupby("Group")["TotalSpend"].transform("mean")
    df["GroupCryoFrac"] = df.groupby("Group")["CryoSleep"].transform("mean")

    # Family-level (surname, cross-group) spend/cryo aggregates
    df["FamilyTotalSpend"] = df.groupby("Surname")["TotalSpend"].transform("sum")
    df["FamilyCryoFrac"] = df.groupby("Surname")["CryoSleep"].transform("mean")
    df["FamilyDeckNunique"] = df.groupby("Surname")["Deck"].transform("nunique")

    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "Deck", "Side",
                "AgeGroup", "CabinRegion"]
    return df, cat_cols


def build_features():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    n_train = len(train)

    full_raw = pd.concat([train, test], ignore_index=True)
    full_fe, cat_cols = engineer(full_raw)
    train_fe = full_fe.iloc[:n_train].reset_index(drop=True)
    test_fe = full_fe.iloc[n_train:].reset_index(drop=True)

    feature_cols = [
        "HomePlanet", "CryoSleep", "Destination", "Age",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "TotalSpend", "LuxurySpend", "LuxuryRatio",
        "RoomService_ratio", "FoodCourt_ratio", "ShoppingMall_ratio", "Spa_ratio", "VRDeck_ratio",
        "GroupSize", "FamilySize",
        "Deck", "CabinNum", "Side", "CabinRegion", "AgeGroup",
        "GroupTotalSpend", "GroupAvgSpend", "GroupCryoFrac",
    ]

    X = train_fe[feature_cols].copy()
    y = train_fe["Transported"].astype(int)
    X_test = test_fe[feature_cols].copy()
    groups = train_fe["Group"]

    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[c].astype(str), X_test[c].astype(str)], axis=0)
        le.fit(combined)
        X[c] = le.transform(X[c].astype(str))
        X_test[c] = le.transform(X_test[c].astype(str))
        encoders[c] = le

    return X, y, X_test, groups, cat_cols, test_fe["PassengerId"]


def load_tuned_params():
    try:
        with open("best_params.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


TUNED = load_tuned_params()


def train_lgb(X_tr, y_tr, X_val, y_val, cat_cols):
    params = dict(
        n_estimators=3000, learning_rate=0.02, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    )
    params.update(TUNED.get("lgb", {}))
    model = lgb.LGBMClassifier(**params, random_state=SEED, verbosity=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_error",
               categorical_feature=cat_cols, callbacks=[lgb.early_stopping(150, verbose=False)])
    return model, model.predict_proba(X_val)[:, 1]


def train_xgb(X_tr, y_tr, X_val, y_val):
    params = dict(
        n_estimators=3000, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        random_state=SEED, eval_metric="logloss", early_stopping_rounds=150,
        verbosity=0,
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model, model.predict_proba(X_val)[:, 1]


def train_cat(X_tr, y_tr, X_val, y_val, cat_cols):
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    params = dict(iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3)
    params.update(TUNED.get("cat", {}))
    model = cb.CatBoostClassifier(
        **params, random_seed=SEED, verbose=False,
        eval_metric="Accuracy", early_stopping_rounds=150,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_idx, use_best_model=True)
    return model, model.predict_proba(X_val)[:, 1]


TE_COLS = ["HomePlanet", "Deck", "Side", "Destination", "CabinRegion", "AgeGroup"]
BAGGING_SEEDS = [42, 7, 123]


def add_target_encoding(X_tr, y_tr, X_val, X_test, te_cols, smoothing=20):
    """Fit smoothed target means on the training fold only, apply to val/test."""
    global_mean = y_tr.mean()
    X_val = X_val.copy()
    X_test = X_test.copy()
    for c in te_cols:
        means = y_tr.groupby(X_tr[c].values).mean()
        counts = y_tr.groupby(X_tr[c].values).count()
        smooth_map = (means * counts + global_mean * smoothing) / (counts + smoothing)
        X_val[f"{c}_te"] = X_val[c].map(smooth_map).fillna(global_mean)
        X_test[f"{c}_te"] = X_test[c].map(smooth_map).fillna(global_mean)
    X_tr = X_tr.copy()
    for c in te_cols:
        # in-fold te for the training rows themselves (mild leakage-safe approx via same smoothing)
        means = y_tr.groupby(X_tr[c].values).mean()
        counts = y_tr.groupby(X_tr[c].values).count()
        smooth_map = (means * counts + global_mean * smoothing) / (counts + smoothing)
        X_tr[f"{c}_te"] = X_tr[c].map(smooth_map).fillna(global_mean)
    return X_tr, X_val, X_test


def main():
    X, y, X_test, groups, cat_cols, test_ids = build_features()

    oof = {m: np.zeros(len(X)) for m in ["lgb", "xgb", "cat"]}
    test_preds = {m: np.zeros(len(X_test)) for m in ["lgb", "xgb", "cat"]}

    for seed_i, seed in enumerate(BAGGING_SEEDS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            X_tr, X_val, X_test_enc = add_target_encoding(X_tr, y_tr, X_val, X_test, TE_COLS)

            lgb_model, lgb_val_pred = train_lgb(X_tr, y_tr, X_val, y_val, cat_cols)
            xgb_model, xgb_val_pred = train_xgb(X_tr, y_tr, X_val, y_val)
            cat_model, cat_val_pred = train_cat(X_tr, y_tr, X_val, y_val, cat_cols)

            if seed_i == 0:
                oof["lgb"][val_idx] = lgb_val_pred
                oof["xgb"][val_idx] = xgb_val_pred
                oof["cat"][val_idx] = cat_val_pred
                for m in ["lgb", "xgb", "cat"]:
                    acc = ((oof[m][val_idx] > 0.5).astype(int) == y_val.values).mean()
                    print(f"Fold {fold} {m}: acc={acc:.4f}")

            n_total = N_FOLDS * len(BAGGING_SEEDS)
            test_preds["lgb"] += lgb_model.predict_proba(X_test_enc)[:, 1] / n_total
            test_preds["xgb"] += xgb_model.predict_proba(X_test_enc)[:, 1] / n_total
            test_preds["cat"] += cat_model.predict_proba(X_test_enc)[:, 1] / n_total

    print()
    for m in ["lgb", "xgb", "cat"]:
        acc = ((oof[m] > 0.5).astype(int) == y.values).mean()
        print(f"OOF {m} accuracy (seed 42 only, for reporting): {acc:.4f}")

    # Simple average blend
    blend_oof = (oof["lgb"] + oof["xgb"] + oof["cat"]) / 3
    blend_acc = ((blend_oof > 0.5).astype(int) == y.values).mean()
    print(f"OOF blend (avg) accuracy: {blend_acc:.4f}")

    # Stacked meta-learner: fit logistic regression on OOF preds, but evaluate
    # via its own inner CV so the reported accuracy isn't in-sample-optimistic.
    meta_X = np.column_stack([oof["lgb"], oof["xgb"], oof["cat"]])
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    stack_oof_pred = cross_val_predict(LogisticRegression(), meta_X, y, cv=inner_cv, method="predict_proba")[:, 1]
    stack_acc = ((stack_oof_pred > 0.5).astype(int) == y.values).mean()
    print(f"OOF stacked accuracy (inner-CV): {stack_acc:.4f}")

    meta = LogisticRegression()
    meta.fit(meta_X, y)
    print(f"Meta weights (fit on full OOF): lgb={meta.coef_[0][0]:.3f}, xgb={meta.coef_[0][1]:.3f}, cat={meta.coef_[0][2]:.3f}")

    # Use whichever strategy scored best on OOF for final submission
    candidates = {
        "lgb": (oof["lgb"], test_preds["lgb"]),
        "xgb": (oof["xgb"], test_preds["xgb"]),
        "cat": (oof["cat"], test_preds["cat"]),
        "blend": (blend_oof, (test_preds["lgb"] + test_preds["xgb"] + test_preds["cat"]) / 3),
    }
    meta_test = meta.predict_proba(np.column_stack([test_preds["lgb"], test_preds["xgb"], test_preds["cat"]]))[:, 1]
    candidates["stack"] = (stack_oof_pred, meta_test)

    best_name = max(candidates, key=lambda k: ((candidates[k][0] > 0.5).astype(int) == y.values).mean())
    best_oof, best_test = candidates[best_name]
    best_acc = ((best_oof > 0.5).astype(int) == y.values).mean()
    print(f"\nBest strategy: {best_name} (OOF acc={best_acc:.4f})")

    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": (best_test > 0.5),
    })
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
