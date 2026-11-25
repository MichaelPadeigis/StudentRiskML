# Final-report experiments

These are stubs for the follow-up work promised in the progress report. Each script:

- reuses `train.py` (through `common.py`), with the **same 80/20 split, seed 42 and 5-fold CV** as the progress report, so the numbers stay comparable;
- writes to `outputs/experiments/<script name>/`;
- stops at a `NotImplementedError` where the work begins, and has `# TODO` comments describing each step.

Run them from the `Student_Risk_ML` folder:

```powershell
.\.venv\Scripts\python.exe experiments\exp1_controlled_comparison.py
```

| Script | What it does | Report section |
|---|---|---|
| `exp1_controlled_comparison.py` | All 4 models × both feature sets, identical settings | Implementation 5.2 |
| `exp2_hyperparameter_tuning.py` | GridSearchCV on the training folds | Implementation 5.1 |
| `exp3_threshold_tuning.py` | Choose the cutoff from out-of-fold predictions, PR curves | Implementation 5.3 / Evaluation |
| `exp4_features_ablation.py` | Zero-approved-units indicator + leave-one-group-out ablation | Features 4.1–4.2 |
| `exp5_three_class.py` | Dropout / Enrolled / Graduate, macro-F1 | Implementation 5.4 / Evaluation |
| `exp6_error_analysis.py` | High-error cases, errors by subgroup, enrollment vs. semester-1 comparison | Error Analysis |

Rule: **only score on the test set once, for the final chosen setting.** Make every decision (model, hyperparameters, cutoff, features) from the CV folds on the training set.

Copy any figure that goes into the report into `../Final_Report/figures/`.
