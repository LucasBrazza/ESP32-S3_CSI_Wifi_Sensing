# Dataset v2 complete experiment suite

This package updates the Dataset v2 branch without replacing the historical
scripts 01–19. The new script consolidates the advanced experiments and adapts
them to the binary Dataset v2 organized by session, quadrant and class.

## Files

- `20_retrain_dataset_v2.py`: complete experiment runner.
- `dataset_v2_training_config.json`: reproducible search spaces, seeds,
  output paths and final-candidate policy.

## Branch preparation

The Dataset v2 branch should contain both:

1. the advanced experiment history from
   `research/literature-guided-pipeline-review`;
2. the binary acquisition and Dataset v2 organization from
   `acquisition/binary-csi-stream`.

From a clean repository:

```powershell
git fetch origin

git switch acquisition/binary-csi-stream
git status

git switch research/dataset-v2-pipeline
git branch backup/dataset-v2-pipeline-before-advanced-sync

git reset --hard origin/research/literature-guided-pipeline-review
git merge --no-commit --no-ff acquisition/binary-csi-stream

# Keep the Dataset directory exactly as organized in the acquisition branch.
# This prevents Dataset v1 results and processed artifacts from returning.
git restore --source=acquisition/binary-csi-stream `
    --staged --worktree -- Tools/datasets

git add -A
git status
git commit -m "merge: combine advanced experiments with Dataset v2 acquisition"
```

If the merge reports conflicts outside `Tools/datasets`, inspect them before
committing. Do not use a global `ours` or `theirs` strategy.

## Installation

The project virtual environment must provide:

```powershell
python -m pip install numpy pandas scipy matplotlib scikit-learn joblib
```

XGBoost is optional:

```powershell
python -m pip install xgboost
```

## Execution sequence

First run a smoke test without XGBoost:

```powershell
python -m Tools.training.20_retrain_dataset_v2 --quick --skip-xgboost
```

Then run the quick suite with XGBoost:

```powershell
python -m Tools.training.20_retrain_dataset_v2 --quick
```

The quick outputs are written separately and must not be reported in the TCC.
Smoke mode skips the expensive binary, hierarchical, session-holdout and
quadrant-holdout stages; all of them remain enabled in the official run.

Run the official suite with:

```powershell
python -m Tools.training.20_retrain_dataset_v2
```

The complete suite can take a long time because it refits preprocessing,
feature selection and classifiers independently for each file-based split.

## Methodological safeguards

- The split unit is the acquisition file, never an isolated window.
- All windows from one file remain entirely in training or testing.
- File holdout is stratified by class, session and quadrant.
- Z-score statistics are fitted only on training files.
- Correlation removal is fitted only on training files.
- Fisher ranking is fitted only on training windows.
- The historical evaluation seeds are retained:
  `7, 13, 21, 42, 84, 126, 168, 210, 336, 512`.

## Experiment coverage

The runner reproduces and expands the advanced branch tests:

1. Dataset integrity, packet count and sampling-rate diagnostics.
2. Correlation-threshold retuning for Dataset v2.
3. Window and step search expressed in seconds.
4. Decision-tree tuning.
5. Broad capacity diagnostic with tree, Random Forest, Extra Trees,
   Gradient Boosting, KNN, linear SVM, RBF SVM and Logistic Regression.
6. Compact ensemble search.
7. Ten-seed classifier stability validation.
8. Professor-suggested Logistic Regression and XGBoost comparison.
9. Gradient Boosting and XGBoost feature-budget comparison.
10. Accuracy, Macro/Weighted F1, Macro/Weighted ROC-AUC,
    Macro/Weighted average precision, per-class metrics and timing.
11. Model-complexity estimates: trees, nodes, used features and comparisons.
12. Per-file, per-session, per-quadrant and class-by-quadrant diagnostics.
13. Empty-versus-presence and static-versus-movement diagnostics.
14. Direct multiclass versus hierarchical classification.
15. Leave-one-session-out and leave-one-quadrant-out validation.
16. Final all-data fit and realtime artifact export.

## Results

Official results are written to:

```text
Tools/datasets/results/dataset_v2_complete_suite/
├── experiment_index.csv
├── figures/
├── logs/
├── reports/
└── tables/
```

The directory contains detailed and summarized CSV files, confusion matrices,
comparison charts and the report:

```text
Tools/datasets/results/dataset_v2_complete_suite/reports/
└── dataset_v2_complete_report.md
```

## Realtime artifacts

Generated binary artifacts are written to the ignored processed directory:

```text
Tools/datasets/processed/
├── realtime_model.joblib
├── realtime_pipeline_config.json
├── realtime_model_structure.json
└── dataset_v2_training_bundle.joblib
```

Two reviewable JSON files are also exported to a versionable path:

```text
Tools/realtime/config/
├── dataset_v2_realtime_pipeline_config.json
└── dataset_v2_realtime_model_structure.json
```

The pipeline JSON records the selected window, preprocessing parameters,
Z-score vectors, retained subcarriers, Fisher-selected features, class order,
classifier parameters, validation summary and serialized-model hash.

The `.joblib` model is required for immediate PC inference. The JSON structure
supports inspection and later conversion to an embedded implementation.

## Suggested commits

Add the experiment implementation before running the official suite:

```powershell
git add Tools/training/20_retrain_dataset_v2.py
git add Tools/training/dataset_v2_training_config.json
git add Tools/training/DATASET_V2_EXPERIMENTS.md
git commit -m "feat(training): add complete Dataset v2 experiment suite"
```

After the official run, inspect the generated results before committing them:

```powershell
git add Tools/datasets/results/dataset_v2_complete_suite
git add Tools/realtime/config/dataset_v2_realtime_pipeline_config.json
git add Tools/realtime/config/dataset_v2_realtime_model_structure.json
git commit -m "results(dataset-v2): add complete training and validation results"
```
