"""EXP 3: Decision cutoff tuning and precision-recall curves.

Report sections: Implementation 5.3, Evaluation, Progress (item 2), Error Analysis 8.1
Progress-report promise: "tune the cutoff on the validation folds and report
precision-recall curves."

Approach:
  1. Out-of-fold probabilities on TRAIN: cross_val_predict(model, X_train, y_train,
     cv=cv_folds(), method='predict_proba')[:, 1]
  2. Pick a cutoff from those, e.g. the one that maximises F1, or the highest
     cutoff that still gives dropout recall >= 0.90 (choose and justify one).
  3. Apply that fixed cutoff to the test set once. Compare with 0.50.

Output: outputs/experiments/exp3_threshold_tuning/
        thresholds.json      chosen cutoff + rule, per timing
        results.csv          test metrics at 0.50 vs tuned cutoff
        pr_curve_<timing>.png   copy the best one into Final_Report/figures/
"""
from common import TIMINGS, build_pipeline, cv_folds, load_split, out_dir

NAME = 'exp3_threshold_tuning'

# TODO: use the tuned model from exp2 once it exists.
MODEL_FOR = {'enrollment': 'logistic_regression', 'after_first_semester': 'svm'}


def main():
    out = out_dir(NAME)
    for timing in TIMINGS:
        X_train, X_test, y_train, y_test = load_split(timing)
        model = build_pipeline(MODEL_FOR[timing], X_train)
        # TODO: oof = cross_val_predict(model, X_train, y_train, cv=cv_folds(), method='predict_proba')[:, 1]
        # TODO: precision, recall, thresholds = precision_recall_curve(y_train, oof); choose cutoff
        # TODO: model.fit(X_train, y_train); evaluate on X_test at 0.50 and at the cutoff
        # TODO: plot PR curve (PrecisionRecallDisplay) and mark the chosen point
        raise NotImplementedError('exp3: fill in threshold selection')


if __name__ == '__main__':
    main()
