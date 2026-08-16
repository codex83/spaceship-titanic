# Spaceship Titanic

Solution for Kaggle's [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition — binary classification predicting which passengers were "transported" to an alternate dimension.

**Public leaderboard score: 0.80383**

## Approach

- **Feature engineering**: group/family extraction from `PassengerId` and `Name`, cabin deck/side/region from `Cabin`, spend ratios and luxury-spend ratio, group-level spend and CryoSleep aggregates, group/family-based imputation for missing `HomePlanet`/`Destination`, CryoSleep imputed from spend patterns (cryosleeping passengers spend nothing).
- **Models**: LightGBM, XGBoost, and CatBoost, hyperparameter-tuned with Optuna (`tune.py`), evaluated with `StratifiedGroupKFold` so passengers from the same travel group never leak across train/validation folds.
- **Target encoding**: smoothed out-of-fold mean-target encoding for the main categorical features, fit per-fold to avoid leakage — the single feature-engineering change with the clearest, reproducible lift.
- **Final submission**: CatBoost, predictions bagged across 3 CV-split seeds for variance reduction. OOF accuracy ≈ 81.9%.

## What didn't work (and why that's informative)

- **Cross-group family aggregates** (spend/CryoSleep patterns linked by surname across travel groups) hurt accuracy — with 2,406 unique surnames across ~12.7k passengers, surname collisions between unrelated families added noise rather than real family signal.
- **Hand-built interaction features** (CryoSleep/spend inconsistency flags, cabin rank within deck, HomePlanet×Deck combos) also hurt.
- **A neural network** (PyTorch, entity embeddings for categoricals) underperformed the tree models (81.2% vs 81.9%) and correlated too highly with CatBoost's predictions (0.97) to add ensemble diversity — expected on a dataset this small (~8.7k rows), where gradient-boosted trees dominate.
- **Ensembling/stacking** the three GBM models never beat tuned CatBoost alone once evaluated honestly (via inner cross-validation rather than in-sample fit).

## Misclassification analysis

Segmenting errors by `HomePlanet × CryoSleep` surfaced a clear, non-random pattern:

| Segment | Accuracy | Transported rate | Share of data |
|---|---|---|---|
| Europa + CryoSleep | 98.8% | 98.8% | 11% |
| Mars + CryoSleep | 91.4% | 90.9% | 8% |
| Earth + not-CryoSleep | 78.9% | 32.1% | 36% |
| **Earth + CryoSleep** | **65.1%** | **65.2%** | **16%** |

For Europa/Mars, CryoSleep is a near-deterministic predictor of the outcome and the model exploits it almost perfectly. For Earth passengers in CryoSleep (16% of the dataset), the outcome is close to a coin flip and the model can barely beat the base rate — checking predicted probabilities against actual rates within that segment shows the model has already extracted the (weak) signal that exists there (e.g. cabin side, destination). This segment is effectively the accuracy ceiling for the whole dataset, consistent with this being a synthetic "Getting Started" competition with deliberately injected irreducible noise.

## On the leaderboard

The public leaderboard's top scores (0.96+) are not representative of genuine modeling — they show classic signatures of leaderboard probing (score-guided label reconstruction via hundreds of submissions) or data leakage (near-perfect scores from a handful of submissions), both well-documented phenomena on unlimited-submission "Getting Started" playground competitions. Honest, well-engineered models on this dataset cluster around 0.80–0.83.

## Files

- `train_model.py` — feature engineering + final training pipeline (LightGBM/XGBoost/CatBoost, target encoding, multi-seed bagging)
- `tune.py` — Optuna hyperparameter search (writes `best_params.json`)
- `experiment.py` — ablation tests for 5 candidate improvements
- `nn_model.py` — PyTorch neural network baseline (entity embeddings) for model-family comparison

Competition data (`data/`) is not included per Kaggle's data redistribution terms — download it from the [competition page](https://www.kaggle.com/competitions/spaceship-titanic/data) and place `train.csv`/`test.csv`/`sample_submission.csv` in a local `data/` folder to reproduce.
