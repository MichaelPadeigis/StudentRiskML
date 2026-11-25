"""EXP 2: Hyperparameter tuning with cross-validation.

Report sections: Implementation 5.1, Evaluation Table 2, Progress (item 3)
Progress-report promise: "tree depth, number of trees and the SVM's C are set
by hand. We will tune them with cross-validation."

Approach: GridSearchCV (or RandomizedSearchCV for RF) on the TRAINING set only,
using cv_folds() and scoring='roc_auc'. Touch the test set once, at the end.
Pipeline step names are 'prep' and 'model', so params look like 'model__C'.

Output: outputs/experiments/exp2_hyperparameter_tuning/
        best_params.json   per (timing, model)
        results.csv        hand-set CV score vs tuned CV score vs tuned test score
"""
from common import MODELS, TIMINGS, build_pipeline, cv_folds, load_split, out_dir

NAME = 'exp2_hyperparameter_tuning'

# TODO: agree on ranges as a group. These are only starting points.
GRIDS = {
    'logistic_regression': {'model__C': [0.01, 0.1, 1, 10]},
    'decision_tree': {'model__max_depth': [3, 5, 7, 10, None],
                      'model__min_samples_leaf': [1, 5, 15, 30]},
    'random_forest': {'model__n_estimators': [200, 500],
                      'model__max_depth': [None, 10, 20],
                      'model__max_features': ['sqrt', 0.5]},
    'svm': {'model__C': [0.1, 1, 10], 'model__gamma': ['scale', 0.01, 0.1]},
}


def main():
    out = out_dir(NAME)
    for timing in TIMINGS:
        X_train, X_test, y_train, y_test = load_split(timing)
        for model_name in MODELS:
            pipe = build_pipeline(model_name, X_train)
            # TODO: search = GridSearchCV(pipe, GRIDS[model_name], cv=cv_folds(),
            #                             scoring='roc_auc', n_jobs=-1)
            # TODO: search.fit(X_train, y_train); record best_params_ and best_score_
            # TODO: evaluate search.best_estimator_ on X_test ONLY for the final chosen model(s)
            raise NotImplementedError('exp2: fill in grid search')
    # TODO: save best_params.json and results.csv into out


if __name__ == '__main__':
    main()
