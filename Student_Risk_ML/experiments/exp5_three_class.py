"""EXP 5: Three-class setup (Dropout / Enrolled / Graduate).

Report sections: Dataset 3.5, Implementation 5.4, Evaluation (Table 1 + results), Progress (item 1)
Progress-report promise: "evaluate a three-class setup with macro-averaged F1
and per-class recall, to show what this exclusion costs."

Note: train.select_data() drops Enrolled students, so this script builds its
own X/y. Keep the same seed, 80/20 split and stratification so it is comparable.

Approach:
  - y = df['Target'] (3 labels); X = same feature columns as the chosen timing
  - build_pipeline(...) works for multiclass (LR becomes multinomial)
  - metrics: macro-F1, per-class recall, a 3x3 confusion matrix
  - fill in the 3-class rows of Table 1 (split sizes and label counts)

Output: outputs/experiments/exp5_three_class/
        split_counts.json, results.csv, confusion_matrix_<timing>.png
"""
from common import RANDOM_STATE, TIMINGS, DATA, build_pipeline, cv_folds, out_dir, read_dataset

NAME = 'exp5_three_class'


def load_three_class(timing):
    df = read_dataset(DATA)
    y = df['Target']
    X = df.drop(columns=['Target'])
    # TODO: drop semester columns exactly as train.select_data does for `timing`
    # TODO: X = X.apply(pd.to_numeric, errors='coerce')
    # TODO: return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    raise NotImplementedError('exp5: build 3-class data')


def main():
    out = out_dir(NAME)
    for timing in TIMINGS:
        X_train, X_test, y_train, y_test = load_three_class(timing)
        # TODO: record split counts (y_train.value_counts(), y_test.value_counts())
        # TODO: CV the 4 models with scoring='f1_macro'; fit the best; evaluate on test
        # TODO: classification_report(y_test, pred, output_dict=True) -> results.csv
        # TODO: ConfusionMatrixDisplay -> confusion_matrix_<timing>.png


if __name__ == '__main__':
    main()
