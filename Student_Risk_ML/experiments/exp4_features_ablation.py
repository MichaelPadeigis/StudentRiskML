"""EXP 4: Engineered features (zero-grade indicator) and feature-group ablations.

Report sections: Features 4.1 + 4.2, Dataset 3.5, Progress (item 4), Error Analysis 8.3
Progress-report promise: "add an indicator for semesters with zero approved
units, so that a 0 is not read as an extremely low grade."
Rubric: "ablations ... include one feature at a time so you can tell what
feature is contributing more to the performance."

Part A - engineered features (after_first_semester only):
  - 'sem1_zero_approved' = (Curricular units 1st sem (approved) == 0)
  - (optional) approved/enrolled ratio, grade set to NaN when zero approved
  Compare CV + test ROC-AUC/F1 with and without them, same model.

Part B - feature-group ablation:
  Drop one group from FEATURE_GROUPS at a time (leave-one-group-out) and
  record the drop in CV ROC-AUC. Optionally also run each group alone.

Output: outputs/experiments/exp4_features_ablation/
        engineered.csv    with vs without engineered features
        ablation.csv      one row per removed group
"""
from common import build_pipeline, cv_folds, load_split, out_dir

NAME = 'exp4_features_ablation'

# Matches Table 1 in the report. TODO: check every column name against data.csv
# (run: python -c "import pandas as pd; print(list(pd.read_csv('data/data.csv', sep=None, engine='python').columns))")
FEATURE_GROUPS = {
    'demographic': ['Gender', 'Age at enrollment', 'Nacionality', 'Displaced',
                    'Marital status', 'International', 'Educational special needs'],
    'socioeconomic': ["Mother's qualification", "Father's qualification",
                      "Mother's occupation", "Father's occupation",
                      'Debtor', 'Tuition fees up to date', 'Scholarship holder'],
    'enrollment': ['Course', 'Application mode', 'Application order',
                   'Daytime/evening attendance', 'Previous qualification',
                   'Previous qualification (grade)', 'Admission grade'],
    'academic_sem1': 'startswith:Curricular units 1st sem',
    'macroeconomic': ['Unemployment rate', 'Inflation rate', 'GDP'],
}


def add_engineered(X):
    """Return a copy of X with engineered columns added. Apply to train and test alike."""
    X = X.copy()
    # TODO: X['sem1_zero_approved'] = (X['Curricular units 1st sem (approved)'] == 0).astype(int)
    # TODO: optional extra features
    return X


def main():
    out = out_dir(NAME)
    X_train, X_test, y_train, y_test = load_split('after_first_semester')
    # TODO Part A: CV (cross_validate, cv=cv_folds(), scoring roc_auc/f1) with and without add_engineered
    # TODO Part B: for each group, drop its columns, rebuild the pipeline with
    #              build_pipeline(name, X_reduced) (it picks categorical columns automatically),
    #              and record the CV score
    raise NotImplementedError('exp4: fill in engineered features + ablation')


if __name__ == '__main__':
    main()
