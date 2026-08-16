# Spaceship Titanic

A solution to Kaggle's [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition, a binary classification task predicting which passengers were transported to an alternate dimension during an anomaly encounter.

**Public leaderboard score: 0.80383**

## Approach

- **Feature engineering** — group and family structure extracted from `PassengerId` and `Name`; cabin deck, side, and region parsed from `Cabin`; per-category spend ratios and a luxury-spend ratio; group-level spend and CryoSleep aggregates; group/family-based imputation for missing `HomePlanet` and `Destination`; CryoSleep imputed from spend patterns, since cryosleeping passengers spend nothing.
- **Models** — LightGBM, XGBoost, and CatBoost, each hyperparameter-tuned with Optuna (`tune.py`) and evaluated with `StratifiedGroupKFold`, so passengers from the same travel group never leak across train/validation folds.
- **Target encoding** — smoothed out-of-fold mean-target encoding for the primary categorical features, fit per fold to avoid leakage. This was the single feature-engineering change with the clearest, most reproducible lift.
- **Final model** — CatBoost, with predictions bagged across three CV-split seeds to reduce variance. Out-of-fold accuracy ≈ 81.9%.

## Approaches that were tested and discarded

- **Cross-group family aggregates** (spend and CryoSleep patterns linked by surname across travel groups) reduced accuracy. With 2,406 unique surnames across roughly 12,700 passengers, surname collisions between unrelated families introduced noise rather than genuine family signal.
- **Hand-built interaction features** — a CryoSleep/spend inconsistency flag, cabin-number rank within deck, and a HomePlanet × Deck combination — also reduced accuracy.
- **A neural network** (PyTorch, entity embeddings for categorical features) underperformed the tree-based models, 81.2% versus 81.9%, and its predictions correlated too highly with CatBoost's (0.97) to add ensemble diversity. This is consistent with expectations for a dataset this size (~8,700 rows), where gradient-boosted trees typically dominate.
- **Ensembling and stacking** the three gradient-boosted models never outperformed tuned CatBoost alone once evaluated honestly, via nested cross-validation rather than an in-sample fit.

## Misclassification analysis

Segmenting errors by `HomePlanet × CryoSleep` reveals a clear, non-random pattern in where the model struggles:

| Segment | Accuracy | Transported rate | Share of data |
|---|---|---|---|
| Europa + CryoSleep | 98.8% | 98.8% | 11% |
| Mars + CryoSleep | 91.4% | 90.9% | 8% |
| Earth + not CryoSleep | 78.9% | 32.1% | 36% |
| Earth + CryoSleep | 65.1% | 65.2% | 16% |

For Europa and Mars, CryoSleep is a near-deterministic predictor of the outcome, and the model exploits it almost perfectly. For Earth passengers in CryoSleep — 16% of the dataset — the outcome is close to even odds, and the model can barely beat the base rate. Comparing predicted probabilities against actual rates within that segment shows the model has already extracted what weak signal exists there (e.g. cabin side, destination). This segment appears to be the effective accuracy ceiling for the dataset as a whole.

## Repository structure

- `train_model.py` — feature engineering and the final training pipeline (LightGBM/XGBoost/CatBoost, target encoding, multi-seed bagging)
- `tune.py` — Optuna hyperparameter search, writes `best_params.json`
- `experiment.py` — ablation tests for candidate feature/modeling improvements
- `nn_model.py` — PyTorch neural network baseline (entity embeddings), used for model-family comparison

Competition data (`data/`) is not included, per Kaggle's data redistribution terms. Download it from the [competition page](https://www.kaggle.com/competitions/spaceship-titanic/data) and place `train.csv`, `test.csv`, and `sample_submission.csv` in a local `data/` folder to reproduce these results.
