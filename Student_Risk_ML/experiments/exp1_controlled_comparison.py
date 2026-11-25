"""EXP 1: Controlled comparison of feature sets.

Report sections: Implementation 5.2, Progress (item 5), Error Analysis 8.4
Progress-report promise: "train the same algorithm on both feature sets to
isolate the effect of first-semester data."

Why: the progress report compared LR at enrollment with an SVM after
semester 1, so the gain mixes up two changes (more data AND a different model).

Output: outputs/experiments/exp1_controlled_comparison/results.csv
        one row per (timing, model) with test ROC-AUC, recall, precision, F1, accuracy.
"""
from common import MODELS, TIMINGS, build_pipeline, load_split, out_dir

NAME = 'exp1_controlled_comparison'


def main():
    out = out_dir(NAME)
    rows = []
    for timing in TIMINGS:
        X_train, X_test, y_train, y_test = load_split(timing)
        for model_name in MODELS:
            model = build_pipeline(model_name, X_train)
            # TODO: model.fit(X_train, y_train)
            # TODO: probability = model.predict_proba(X_test)[:, 1]; prediction = probability >= 0.5
            # TODO: append a dict with timing, model, roc_auc, recall, precision, f1, accuracy
            #       (sklearn.metrics; copy the metric calls from train.py train_one)
            raise NotImplementedError('exp1: fill in fit/evaluate')
    # TODO: pd.DataFrame(rows).to_csv(out / 'results.csv', index=False)
    # TODO (optional): pivot to a 4x2 table and/or bar chart -> Final_Report/figures/


if __name__ == '__main__':
    main()
