"""Shared helpers for the final-report experiments.

Every experiment reuses train.py so that the split, preprocessing and seed
match the progress report exactly. Do not create a new split anywhere else,
or the results stop being comparable with Table 2 of the report.

Run experiments from the Student_Risk_ML folder, e.g.
    .venv\\Scripts\\python.exe experiments\\exp1_controlled_comparison.py
"""
import json
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]          # Student_Risk_ML/
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402  (import after sys.path change)
    RANDOM_STATE, TIMINGS, build_pipeline, read_dataset, select_data,
)

DATA = ROOT / 'data' / 'data.csv'
OUT = ROOT / 'outputs' / 'experiments'
MODELS = ['logistic_regression', 'decision_tree', 'random_forest', 'svm']


def load_split(timing='enrollment'):
    """Return X_train, X_test, y_train, y_test using the progress-report split."""
    X, y = select_data(read_dataset(DATA), timing)
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def cv_folds():
    """The same 5-fold stratified CV used in train.py."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def out_dir(name):
    """outputs/experiments/<name>/, created if missing."""
    path = OUT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding='utf-8')


__all__ = [
    'RANDOM_STATE', 'TIMINGS', 'MODELS', 'DATA', 'build_pipeline', 'read_dataset',
    'select_data', 'load_split', 'cv_folds', 'out_dir', 'save_json',
]
