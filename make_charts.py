import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_model import build_features

X, y, X_test, groups, cat_cols, test_ids = build_features()
yv = y.values


def acc(p):
    return ((p > 0.5).astype(int) == yv).mean()


cat_oof = np.load("assets_cat_oof.npy")
lgb_oof = np.load("assets_lgb_oof.npy")
xgb_oof = np.load("assets_xgb_oof.npy")
nn_oof = np.load("nn_oof.npy")

with open("assets_feat_imp.json") as f:
    feat_imp_raw = json.load(f)

train_raw = pd.read_csv("data/train.csv")

PALETTES = {
    "light": dict(
        suffix="", text="#333333", grid="#E5E5E5", accent="#4C72B0",
        accent2="#DD8452", highlight="#C44E52", muted="#888888",
    ),
    "dark": dict(
        suffix="_dark", text="#E6EDF3", grid="#30363D", accent="#6FA8DC",
        accent2="#E8A458", highlight="#F0666E", muted="#8B949E",
    ),
}


def render_all(theme):
    p = PALETTES[theme]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.edgecolor": p["muted"],
        "axes.labelcolor": p["text"],
        "text.color": p["text"],
        "xtick.color": p["text"],
        "ytick.color": p["text"],
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
    })

    # --- Chart 1: model comparison ---
    models = ["XGBoost", "Neural Net\n(embeddings)", "LightGBM", "CatBoost\n(final)"]
    scores = [acc(xgb_oof), acc(nn_oof), acc(lgb_oof), acc(cat_oof)]
    colors = [p["accent"]] * 3 + [p["accent2"]]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.barh(models, [s * 100 for s in scores], color=colors, height=0.55, zorder=3)
    ax.set_xlim(78, 83)
    ax.set_xlabel("Out-of-fold accuracy (%)")
    ax.set_title("Model comparison (group-aware 5-fold CV)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", color=p["grid"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{s*100:.2f}%", va="center", fontsize=11, fontweight="medium", color=p["text"])
    plt.tight_layout()
    plt.savefig(f"assets/model_comparison{p['suffix']}.png", dpi=180, transparent=True)
    plt.close()

    # --- Chart 2: feature importance ---
    feat_imp = feat_imp_raw[:12][::-1]
    names = [n for n, _ in feat_imp]
    vals = [v for _, v in feat_imp]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(names, vals, color=p["accent"], height=0.6, zorder=3)
    ax.set_xlabel("CatBoost feature importance")
    ax.set_title("Top predictive features", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", color=p["grid"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"assets/feature_importance{p['suffix']}.png", dpi=180, transparent=True)
    plt.close()

    # --- Chart 3: misclassification by segment ---
    df = X.copy()
    df["y_true"] = yv
    df["correct"] = ((cat_oof > 0.5).astype(int) == yv)
    df["HomePlanet_raw"] = train_raw["HomePlanet"]
    df["CryoSleep_raw"] = train_raw["CryoSleep"]

    seg_labels = ["Europa +\nCryoSleep", "Mars +\nCryoSleep", "Earth + not\nCryoSleep", "Earth +\nCryoSleep"]
    seg_masks = [
        (df.HomePlanet_raw == "Europa") & (df.CryoSleep_raw == True),
        (df.HomePlanet_raw == "Mars") & (df.CryoSleep_raw == True),
        (df.HomePlanet_raw == "Earth") & (df.CryoSleep_raw == False),
        (df.HomePlanet_raw == "Earth") & (df.CryoSleep_raw == True),
    ]
    seg_acc = [df[m]["correct"].mean() * 100 for m in seg_masks]
    seg_n = [m.sum() for m in seg_masks]
    seg_colors = [p["accent"], p["accent"], p["accent"], p["highlight"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(seg_labels, seg_acc, color=seg_colors, width=0.55, zorder=3)
    overall = acc(cat_oof) * 100
    ax.axhline(overall, color=p["muted"], linestyle="--", linewidth=1, zorder=2)
    ax.text(3.6, overall + 0.8, "overall avg", fontsize=9, color=p["muted"], ha="right")
    ax.set_ylim(50, 105)
    ax.set_ylabel("Model accuracy (%)")
    ax.set_title("Accuracy by HomePlanet × CryoSleep segment", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", color=p["grid"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for bar, s, n in zip(bars, seg_acc, seg_n):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{s:.1f}%\n(n={n})", ha="center", fontsize=10, color=p["text"])
    plt.tight_layout()
    plt.savefig(f"assets/misclassification_segments{p['suffix']}.png", dpi=180, transparent=True)
    plt.close()


for theme in ["light", "dark"]:
    render_all(theme)

print("Saved light + dark charts to assets/")
