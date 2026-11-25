# Final Report Plan (Group 13)

**Due: Thursday, December 10, 2026**, submitted to A2L as one zip (report PDF + code + data).

## What the assignment asks for

- A PDF built from the **same ACL LaTeX template** as the progress report. Extra packages are fine, but don't change the formatting much.
- **At least 6 pages of text, not counting figures and tables.** If a figure takes up space, write more to make up for it.
- Start from the progress report. **Remove "Feedback and Plans"** and **add "Progress" and "Error Analysis"**.
- **At least 7 references, with at least 5 published studies**, discussed in the text rather than just listed.
- **Submission zip:** the report PDF, the code, and `data.csv` (0.8 MB, under the 50 MB limit, so include it).

## Sections, weights and owners

Owners are blank on purpose, so fill them in as a group. The **Exp** column lists the experiment scripts that feed each section (see the next table).

| # | Section | Weight | Starting point | What's left | Exp | Owner | Status |
|---|---|---:|---|---|---|---|---|
| 1 | Introduction + Related Work | 10% | Copied from PR §1–2 | Expand the problem statement, tie each paper to our choices, say whether related work changed our direction, maybe add 1–2 references | – | | Not started |
| 2 | Dataset | 5% | Copied from PR §3 | Add "Changes since the progress report" (3-class data, engineered features) | 4, 5 | | Not started |
| 3 | Features and Inputs | 5% | Copied from PR §4 | Engineered features and why; how inputs were varied (ablations) | 4 | | Not started |
| 4 | Implementation | 20% | Copied from PR §5 | Tuning, controlled comparison, cutoff, 3-class model; name the trained baseline we beat | 1, 2, 3, 5 | | Not started |
| 5 | Evaluation | 15% | Copied from PR §6; numbers moved into Table 2 | Fill the split table and results table; discuss whether our metrics were adequate | 1, 2, 3, 5 | | Not started |
| 6 | Progress | 10% | **New**: the 5 PR plan items are listed | One paragraph per item: done? what we found? did the plan change, and why? | all | | Not started |
| 7 | Error Analysis | 20% | **New**: confusion matrices already in | High-error cases, errors by subgroup, differences between models, fixes | 6 (+3, 5) | | Not started |
| 8 | Team Contributions | 5% | PR version kept as a comment | Rewrite to cover the whole project, at least one sentence per member, no "all contributed equally" | – | | Not started |

Status values: Not started / Drafting / Needs review / Done

## Experiments

The scripts are in [`Student_Risk_ML/experiments/`](../Student_Risk_ML/experiments/README.md), and each one already has its own split loading and output folder. Each script stops at a `NotImplementedError` where the work begins.

| Exp | Script | Progress-report promise | Feeds |
|---|---|---|---|
| 1 | `exp1_controlled_comparison.py` | Same algorithm on both feature sets | Impl 5.2, Progress, Error 8.4 |
| 2 | `exp2_hyperparameter_tuning.py` | Tune depth, number of trees and C with cross-validation | Impl 5.1, Table 2 |
| 3 | `exp3_threshold_tuning.py` | Tune the 0.50 cutoff, PR curves | Impl 5.3, Eval, Error 8.1 |
| 4 | `exp4_features_ablation.py` | Zero-grade indicator (+ feature-group ablations, which the rubric asks for) | Features 4.1–4.2, Dataset |
| 5 | `exp5_three_class.py` | 3-class setup, macro-F1, per-class recall | Dataset, Impl 5.4, Eval, Table 1 |
| 6 | `exp6_error_analysis.py` | – (needed for the 20% Error Analysis section) | Error 8.1–8.4 |

| Exp | Owner | Status |
|---|---|---|
| 1 | | Not started |
| 2 | | Not started |
| 3 | | Not started |
| 4 | | Not started |
| 5 | | Not started |
| 6 | | Not started |

**Suggested order:** 2 (tuning) → 1 and 3 (use the tuned models) → 4 and 5 (independent) → 6 last, on the final model's predictions.

## How the .tex is marked up

- <span style="color:blue">**[COPIED: ...]**</span> (blue, `\copynote{}`): this text came from the progress report, and the note says which section. Each copied section also has a `% COPIED FROM PROGRESS REPORT` comment block in the source.
- <span style="color:red">**[TODO: ...]**</span> (red, `\todo{}`): work still to do. Most TODOs restate the rubric questions for that section.
- Each section starts with a `% RUBRIC (x%)` comment that restates what the instructions require.
- **Before submitting:** change `\draftnotestrue` to `\draftnotesfalse` near the top of `final_report.tex`, then search for `\todo` to be sure none are left.

## Build

```bash
cd Final_Report
latexmk -pdf final_report.tex
```

The compiled PDF is gitignored. Put new figures in `Final_Report/figures/` and include them with `\includegraphics{name.png}`.

## Submission checklist

- [ ] All `\todo` removed and `\draftnotesfalse` set
- [ ] At least 6 pages of text, not counting figures and tables
- [ ] At least 7 references, at least 5 of them published studies, every one cited in the text
- [ ] Majority baseline and at least one trained baseline reported; our model beats them, or we explain why not
- [ ] Loss function and optimizer described for each model
- [ ] Split sizes and label distributions reported (Table 1)
- [ ] Error Analysis has figures, statistics, examples and future fixes
- [ ] Team Contributions has a specific sentence for every member
- [ ] "Feedback and Plans" section removed (already done in the template)
- [ ] Zip: `final_report.pdf` + `Student_Risk_ML/` (code, `data/data.csv`, outputs) + `Final_Report/` sources
- [ ] Zip uploaded to A2L
