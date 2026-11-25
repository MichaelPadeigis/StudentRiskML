"""EXP 6: Systematic error analysis (feeds the 20% Error Analysis section).

Report sections: Error Analysis 8.1-8.4
Input: outputs/<timing>/test_predictions.csv, written by train.py. It holds
       every test feature + actual_dropout + dropout_probability + predicted_dropout.
       Re-run train.py (or point at a tuned model's predictions) if the final model changes.

Steps:
  1. high_error_cases.csv: the most confident wrong predictions,
     error = |actual_dropout - dropout_probability|, top N per timing.
  2. errors_by_subgroup.csv: for each subgroup column, n, dropout rate, recall,
     precision, FN rate, FP rate. Suggested columns: Course, age band
     (<=20, 21-25, 26-30, >30), sem1 zero grade, Scholarship holder, Debtor,
     Tuition fees up to date, Gender.
  3. model_diff.csv: join enrollment and after_first_semester predictions on
     the test students (same split, so align by row order or index) and
     categorise them: caught by both, only caught after sem 1, missed by both, etc.

Output: outputs/experiments/exp6_error_analysis/
"""
import pandas as pd

from common import ROOT, TIMINGS, out_dir

NAME = 'exp6_error_analysis'
TOP_N = 10


def load_predictions(timing):
    return pd.read_csv(ROOT / 'outputs' / timing / 'test_predictions.csv')


def main():
    out = out_dir(NAME)
    preds = {t: load_predictions(t) for t in TIMINGS}
    # TODO 1: for each timing, df['error'] = (df.actual_dropout - df.dropout_probability).abs();
    #         keep the wrong rows, sort by error, take TOP_N
    # TODO 2: group by subgroup columns -> error/recall stats
    # TODO 3: compare preds['enrollment'] vs preds['after_first_semester'] row by row
    raise NotImplementedError('exp6: fill in error analysis')


if __name__ == '__main__':
    main()
