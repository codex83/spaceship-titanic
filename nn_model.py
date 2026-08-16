import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from train_model import build_features, SEED, N_FOLDS

CAT_COLS_EMBED = ["HomePlanet", "CryoSleep", "Destination", "Deck", "Side", "AgeGroup", "CabinRegion"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularNN(nn.Module):
    def __init__(self, cat_cardinalities, n_cont):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(card, min(50, (card + 1) // 2 + 1)) for card in cat_cardinalities
        ])
        emb_dim = sum(e.embedding_dim for e in self.embeds)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        in_dim = emb_dim + n_cont
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat, x_cont):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(embs + [self.bn_cont(x_cont)], dim=1)
        return self.net(x).squeeze(1)


def train_nn_fold(X_tr, y_tr, X_val, y_val, X_test, cat_cols, cont_cols, cat_cardinalities, epochs=60, patience=8):
    torch.manual_seed(SEED)

    scaler = StandardScaler()
    Xtr_cont = scaler.fit_transform(X_tr[cont_cols].values.astype(np.float32))
    Xval_cont = scaler.transform(X_val[cont_cols].values.astype(np.float32))
    Xtest_cont = scaler.transform(X_test[cont_cols].values.astype(np.float32))

    Xtr_cat = X_tr[cat_cols].values.astype(np.int64)
    Xval_cat = X_val[cat_cols].values.astype(np.int64)
    Xtest_cat = X_test[cat_cols].values.astype(np.int64)

    tr_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xtr_cat), torch.tensor(Xtr_cont, dtype=torch.float32),
        torch.tensor(y_tr.values, dtype=torch.float32),
    )
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=128, shuffle=True)

    Xval_cat_t = torch.tensor(Xval_cat).to(DEVICE)
    Xval_cont_t = torch.tensor(Xval_cont, dtype=torch.float32).to(DEVICE)
    Xtest_cat_t = torch.tensor(Xtest_cat).to(DEVICE)
    Xtest_cont_t = torch.tensor(Xtest_cont, dtype=torch.float32).to(DEVICE)

    model = TabularNN(cat_cardinalities, len(cont_cols)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb_cat, xb_cont, yb in tr_loader:
            xb_cat, xb_cont, yb = xb_cat.to(DEVICE), xb_cont.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb_cat, xb_cont)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(Xval_cat_t, Xval_cont_t)
            val_loss = loss_fn(val_out, torch.tensor(y_val.values, dtype=torch.float32).to(DEVICE)).item()
        sched.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(model(Xval_cat_t, Xval_cont_t)).cpu().numpy()
        test_pred = torch.sigmoid(model(Xtest_cat_t, Xtest_cont_t)).cpu().numpy()
    return val_pred, test_pred


def main():
    X, y, X_test, groups, cat_cols, test_ids = build_features()

    cont_cols = [c for c in X.columns if c not in CAT_COLS_EMBED]
    cat_cardinalities = [int(X[c].max()) + 1 for c in CAT_COLS_EMBED]  # +1 for 0-index

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        val_pred, t_pred = train_nn_fold(X_tr, y_tr, X_val, y_val, X_test,
                                           CAT_COLS_EMBED, cont_cols, cat_cardinalities)
        oof[val_idx] = val_pred
        test_pred += t_pred / N_FOLDS

        acc = ((val_pred > 0.5).astype(int) == y_val.values).mean()
        print(f"Fold {fold} nn: acc={acc:.4f}")

    oof_acc = ((oof > 0.5).astype(int) == y.values).mean()
    print(f"\nOOF nn accuracy: {oof_acc:.4f}")

    np.save("nn_oof.npy", oof)
    np.save("nn_test.npy", test_pred)
    print("Saved nn_oof.npy / nn_test.npy")


if __name__ == "__main__":
    main()
