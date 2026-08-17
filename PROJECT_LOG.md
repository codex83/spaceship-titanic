# Project Log: Spaceship Titanic

A narrative record of how this project actually unfolded — the decisions, the dead ends, and the corrections along the way. The polished result lives in [README.md](README.md); this document is the story behind it.

## 1. Getting started

The project began with a simple goal: participate in Kaggle's [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition. The Kaggle CLI and credentials were already configured on the machine, so the first step was straightforward — download `train.csv`, `test.csv`, and `sample_submission.csv`, and inspect the data. It's a standard tabular binary classification problem: 8,693 training passengers, 14 columns, a roughly balanced target (`Transported`), and moderate missingness scattered across most features.

## 2. First baseline

A baseline pipeline went up quickly: parse `PassengerId` into travel groups, split `Cabin` into deck/number/side, sum spending across the five amenity columns, impute `CryoSleep` from the fact that cryosleeping passengers spend nothing, and train a 5-fold LightGBM model. It scored **~81.5% out-of-fold accuracy** — a solid, unremarkable first pass.

## 3. A correction: "you didn't take my input"

After the first result, the user pushed back — the model and feature choices had all been made unilaterally, without asking what they wanted. This was a fair and important correction. From that point on, model-family and feature-engineering-depth decisions were put to the user directly (via explicit questions) rather than assumed. The user chose to try all three major gradient-boosting libraries and to go deep on feature engineering rather than stay minimal.

## 4. Building out the pipeline

This produced the shape the project would keep for most of its life:

- **Feature engineering**: group size, family size (via surname), cabin deck/side/region, per-category spend ratios, luxury-spend ratio, group-level spend and CryoSleep aggregates, and group-based imputation for missing `HomePlanet`/`Destination`.
- **Models**: LightGBM, XGBoost, and CatBoost, evaluated with `StratifiedGroupKFold` so passengers from the same travel group never leaked across folds — an easy mistake to make with this dataset, since group members' outcomes are correlated.
- **Ensembling**: blending and stacking were tried, evaluated honestly via inner cross-validation rather than an in-sample fit.

This iteration landed around **81.5–81.9% OOF accuracy**, with CatBoost usually leading.

## 5. Pushing further: tuning and pruning

Told to keep pushing, the next round added Optuna hyperparameter search for LightGBM and CatBoost, and pruned features that showed near-zero importance (`VIP`, `HasSpend`, `IsAlone`, `IsNameMissing`). It also caught and fixed a subtle earlier mistake: the stacking meta-learner's reported accuracy had been computed by fitting and scoring it on the same out-of-fold predictions, which is mildly optimistic. Re-evaluating with proper inner cross-validation showed stacking never actually beat plain tuned CatBoost — a useful negative result. This phase settled around **81.8–81.9%**.

## 6. The "NLP angle" that didn't work

Asked to keep pushing and specifically to explore an NLP-style angle on passenger names, the honest answer was that these are synthetic names with no semantic content — the only real signal available is surname-based family linkage. That was tested directly: cross-group family aggregates (family total spend, family CryoSleep fraction, family deck diversity, linked by surname across travel groups). It **hurt** accuracy. With 2,406 unique surnames across ~12,700 passengers, surname collisions between unrelated families introduced noise rather than genuine family signal. The features were reverted, keeping only `FamilySize` and the (safer) surname-based imputation fallback.

## 7. First submission, and a leaderboard reality check

The user submitted manually and got **0.80383** on the public leaderboard — closely matching the cross-validated estimate. They also flagged that the top of the leaderboard sat at 0.96516. This warranted an honest explanation rather than false modesty: on unlimited-submission "Getting Started" playground competitions like this one, top scores are typically not the product of better modeling. When the user later shared a leaderboard screenshot, the entry-count pattern confirmed it directly — rank 1 scored 0.965 using only 3 submissions (consistent with a data leak, since no iteration is needed if you already have the answers), while ranks 2–4 scored ~0.89 using 450–610 submissions each (consistent with leaderboard probing, where repeated submissions are used to reverse-engineer individual test labels from score feedback). Honest, well-engineered models on this dataset cluster in the 0.80–0.83 range — which is exactly where this one landed.

## 8. Five improvement ideas, tested rigorously

Asked to act as an "expert data scientist" and find five ways to improve the result, each idea was implemented and measured in isolation against the same baseline, rather than bundled together and hoped for the best:

1. **Out-of-fold target encoding** for the main categorical features — **+0.14pp**, kept.
2. **Log1p transforms** of skewed spend features — +0.08pp alone, but discarded once combined with target encoding, since combining the two actually performed *worse* than either alone (redundant signal, added noise).
3. **Hand-built interaction features** (CryoSleep/spend inconsistency flag, cabin rank, HomePlanet × Deck) — hurt accuracy, discarded.
4. **Multi-seed CV bagging** — didn't move the OOF metric, but was kept for the final submission anyway, since it reduces prediction variance on the real test set even when it doesn't show up in-sample.
5. **Decision threshold tuning** — confirmed 0.5 was already optimal.

Target encoding was wired permanently into the training loop (fit per fold, to avoid leakage), and the final submission's predictions were bagged across three CV-split seeds. This pushed CatBoost to **81.94% OOF accuracy**, the best result of the project.

## 9. A different model family, and where the model actually fails

Two more directions were requested together: try a genuinely different model family, and dig into the misclassified rows for a pattern.

A PyTorch neural network with entity embeddings for the categorical features was built and evaluated under the same CV. It underperformed the tree models (81.2% vs. 81.9%) and, more importantly, its predictions correlated 0.97 with CatBoost's — too little diversity to help any ensemble. This matches expectations for a dataset this small (~8,700 rows), where gradient-boosted trees typically dominate over neural networks.

The misclassification analysis was more revealing. Segmenting errors by `HomePlanet × CryoSleep` showed a sharp, non-random pattern: for Europa and Mars, CryoSleep is a near-deterministic predictor of the outcome, and the model captures that almost perfectly (91–99% accuracy in those segments). But for Earth passengers in CryoSleep — 16% of the entire dataset — the outcome is close to a coin flip, and the model can barely beat the base rate (65.1% accuracy). Checking predicted probabilities against actual rates within that segment confirmed the model wasn't underfitting; it had already extracted what weak signal exists (e.g. cabin side). This segment appears to be the effective accuracy ceiling for the dataset as a whole.

## 10. Stopping point

The user's read on this was exactly right: the dataset appears intentionally constructed so that a chunk of it is close to irreducible noise, which is common for Kaggle's synthetic "Getting Started" competitions — enough deterministic structure to reward good modeling, plus enough injected randomness that no model can approach 100%, and where the leaderboard's true top scores come from exploits rather than better data science. With that shared understanding, active modeling work stopped here, at **0.80383 public / ~81.9% OOF**.

## 11. Publishing it

The project moved to GitHub as `codex83/spaceship-titanic`, public, MIT-licensed. Competition data was excluded from the repo per Kaggle's redistribution terms. Two corrections happened during this phase, both acted on immediately:

- A `Co-Authored-By: Claude` trailer in the first commit caused Claude to be listed as a GitHub contributor. Amending the commit and force-pushing didn't resolve it to the user's satisfaction, so the repository was deleted (manually, via GitHub's settings, since the automation's token lacked the `delete_repo` scope and requesting it would have permanently over-broadened its permissions for a one-time task) and recreated from scratch with a single, cleanly authored commit.
- The initial README was rated 6/10. It went through several rounds of real revision: a first pass adding structure and real (not mock) charts generated from the session's actual cross-validation runs, an "On the leaderboard" section that was later cut for tone, an MIT license, and a pipeline diagram.

## 12. The pipeline diagram saga

The diagram turned out to be the most iterated single piece of the whole project:

1. A first version used a single-line Mermaid flowchart — rated hard to read.
2. Rebuilt as a 3-row, 3-box-per-row Mermaid diagram using subgraphs — an improvement, but the connecting arrows between rows landed in the visual center rather than lining up with the actual boxes they connected, because GitHub's underlying dagre layout engine centers edges between subgraphs rather than tracking individual node positions.
3. Reversing the middle row's internal direction (to attempt a "snake" layout) was tried next and confirmed, via a locally rendered test with the same Mermaid engine, that the misalignment persisted regardless of node order — a genuine limitation of the layout engine, not a syntax mistake.
4. The diagram was rebuilt entirely as a static image with hand-placed coordinates (matplotlib), giving exact control over where every arrow starts and ends. This fixed the alignment completely.
5. That image was then found to have two more issues, both fixed in quick succession: the leftmost column's box borders were clipped by the axis limits, and — more importantly — the entire canvas had an opaque white background, which rendered as a jarring white rectangle on GitHub's dark theme.
6. Fixing the background led to checking whether the *other* three chart images had the same problem. They did — `make_charts.py` had hardcoded white backgrounds throughout. All four images (three charts plus the pipeline diagram) were regenerated with true light and dark variants, verified by compositing each one against GitHub's actual dark background color rather than assuming, and wired into the README via `<picture>` tags with `prefers-color-scheme` so the right variant loads automatically for each viewer.

A table of contents was added alongside this pass. The README was rated 9/10 at this point, and the project was considered complete.

## Final state

| | |
|---|---|
| Public leaderboard score | 0.80383 |
| Out-of-fold CV accuracy | 81.94% |
| Final model | CatBoost, tuned with Optuna, 3-seed bagged |
| Repository | [github.com/codex83/spaceship-titanic](https://github.com/codex83/spaceship-titanic) |

The project's real value ended up being less about squeezing out marginal accuracy points and more about the discipline applied along the way: honest cross-validation, ideas tested in isolation before being trusted, negative results kept and reported rather than quietly dropped, and a clear-eyed account of both the dataset's real ceiling and the leaderboard's fake one.
