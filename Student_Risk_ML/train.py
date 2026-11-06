import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET = 'Target'
RANDOM_STATE = 42
CATEGORICAL = [
    'Marital status', 'Application mode', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Displaced',
    'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Gender', 'Scholarship holder', 'International'
]
TIMINGS = ['enrollment', 'after_first_semester']


def read_dataset(path):
    df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    if TARGET not in df.columns:
        raise ValueError('The dataset needs a Target column')
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    if not set(df[TARGET]).issubset({'dropout', 'graduate', 'enrolled'}):
        raise ValueError('Unknown Target values')
    return df


def select_data(df, timing='enrollment'):
    if timing not in TIMINGS:
        raise ValueError('Unknown prediction timing')
    final = df.loc[df[TARGET].isin(['dropout', 'graduate'])].copy()
    y = final[TARGET].eq('dropout').astype(int)
    X = final.drop(columns=[TARGET]).copy()
    if timing == 'enrollment':
        X = X.loc[:, ~X.columns.str.startswith(('Curricular units 1st sem', 'Curricular units 2nd sem'))]
    else:
        X = X.loc[:, ~X.columns.str.startswith('Curricular units 2nd sem')]
    X = X.apply(pd.to_numeric, errors='coerce')
    return X, y


def build_pipeline(name, X):
    categorical = [col for col in CATEGORICAL if col in X.columns]
    numeric = [col for col in X.columns if col not in categorical]
    prep = ColumnTransformer([
        ('categorical', Pipeline([
            ('fill', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical),
        ('numeric', Pipeline([
            ('fill', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), numeric)
    ])
    choices = {
        'logistic_regression': LogisticRegression(max_iter=3000, class_weight='balanced', random_state=RANDOM_STATE),
        'decision_tree': DecisionTreeClassifier(max_depth=7, min_samples_leaf=15, class_weight='balanced', random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=2),
        'svm': SVC(C=1, probability=True, class_weight='balanced', random_state=RANDOM_STATE)
    }
    return Pipeline([('prep', prep), ('model', choices[name])])


def risk_level(probability, medium=0.25, high=0.60):
    if not 0 <= medium < high <= 1:
        raise ValueError('Risk thresholds must be increasing between 0 and 1')
    if probability >= high:
        return 'High'
    if probability >= medium:
        return 'Moderate'
    return 'Low'


def plot_results(path, y_test, probability, prediction, comparison, model, X_train, y_train):
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, prediction, display_labels=['Graduate', 'Dropout'],
        ax=ax, cmap='Blues', colorbar=False
    )
    ax.set_title('Held-out test confusion matrix')
    fig.tight_layout()
    fig.savefig(path / 'confusion_matrix.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5))
    RocCurveDisplay.from_predictions(y_test, probability, ax=ax, name='Chosen model')
    ax.set_title('Held-out test ROC curve')
    fig.tight_layout()
    fig.savefig(path / 'roc_curve.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5))
    precision, recall, _ = precision_recall_curve(y_test, probability)
    ax.plot(recall, precision, linewidth=2)
    ax.axhline(y_test.mean(), linestyle='--', linewidth=1, label='Class rate')
    ax.set(xlabel='Recall', ylabel='Precision', title='Held-out test precision-recall curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'precision_recall.png', dpi=160)
    plt.close(fig)

    display = comparison.sort_values('cv_roc_auc')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(display['model'], display['cv_roc_auc'])
    ax.set(xlabel='Mean cross-validation ROC-AUC', title='Model comparison (training folds only)')
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path / 'model_comparison.png', dpi=160)
    plt.close(fig)

    subset = X_train.sample(n=min(500, len(X_train)), random_state=RANDOM_STATE)
    importance = permutation_importance(
        model, subset, y_train.loc[subset.index],
        scoring='roc_auc', n_repeats=3, random_state=RANDOM_STATE, n_jobs=1
    )
    top = pd.DataFrame({
        'feature': X_train.columns,
        'mean_importance': importance.importances_mean,
        'std_importance': importance.importances_std
    }).sort_values('mean_importance', ascending=False)
    top.to_csv(path / 'feature_importance.csv', index=False)
    display = top.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(display['feature'], display['mean_importance'])
    ax.set(xlabel='ROC-AUC decrease when feature is shuffled', title='Permutation importance (training sample)')
    fig.tight_layout()
    fig.savefig(path / 'feature_importance.png', dpi=160)
    plt.close(fig)


def train_one(df, timing, root):
    out = root / timing
    out.mkdir(parents=True, exist_ok=True)
    X, y = select_data(df, timing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    names = ['logistic_regression', 'decision_tree', 'random_forest', 'svm']
    rows = []
    for name in names:
        print(f'[{timing}] checking {name}', flush=True)
        scores = cross_validate(
            build_pipeline(name, X_train), X_train, y_train, cv=cv,
            scoring={
                'roc_auc': 'roc_auc', 'average_precision': 'average_precision',
                'recall': 'recall', 'precision': 'precision',
                'f1': 'f1', 'accuracy': 'accuracy'
            }, n_jobs=1, error_score='raise'
        )
        row = {'model': name}
        for metric in ['roc_auc', 'average_precision', 'recall', 'precision', 'f1', 'accuracy']:
            row[f'cv_{metric}'] = float(scores[f'test_{metric}'].mean())
            row[f'cv_{metric}_std'] = float(scores[f'test_{metric}'].std())
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values('cv_roc_auc', ascending=False)
    comparison.to_csv(out / 'cross_validation.csv', index=False)

    winner = str(comparison.iloc[0]['model'])
    print(f'[{timing}] best model: {winner}', flush=True)
    model = build_pipeline(winner, X_train)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    cm = confusion_matrix(y_test, prediction, labels=[0, 1])
    metrics = {
        'timing': timing,
        'chosen_model': winner,
        'choice_metric': 'mean 5-fold training ROC-AUC',
        'rows_total': int(len(df)),
        'rows_modelled': int(len(X)),
        'excluded_enrolled': int(len(df) - len(X)),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'feature_count': len(X.columns),
        'features': X.columns.tolist(),
        'dropout_count': int(y.sum()),
        'graduate_count': int((1 - y).sum()),
        'test_dropout_count': int(y_test.sum()),
        'test_accuracy': float(accuracy_score(y_test, prediction)),
        'test_precision': float(precision_score(y_test, prediction)),
        'test_recall': float(recall_score(y_test, prediction)),
        'test_f1': float(f1_score(y_test, prediction)),
        'test_roc_auc': float(roc_auc_score(y_test, probability)),
        'test_average_precision': float(average_precision_score(y_test, probability)),
        'test_brier_score': float(brier_score_loss(y_test, probability)),
        'test_confusion_matrix': cm.tolist(),
        'binary_threshold': 0.50,
        'risk_thresholds': {'moderate': 0.25, 'high': 0.60},
        'note': 'Risk categories are illustrative. This predicts observed dropout, not verified academic support needs.'
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    joblib.dump({
        'pipeline': model, 'features': X.columns.tolist(), 'timing': timing,
        'median_inputs': X_train.median().to_dict(), 'model_name': winner
    }, out / 'model.joblib', compress=3)

    results = X_test.copy()
    results['actual_dropout'] = y_test.to_numpy()
    results['dropout_probability'] = probability
    results['predicted_dropout'] = prediction
    results['support_priority'] = [risk_level(p) for p in probability]
    results.to_csv(out / 'test_predictions.csv', index=False)
    X_test.sample(n=min(15, len(X_test)), random_state=RANDOM_STATE).to_csv(out / 'demo_students.csv', index=False)
    plot_results(out, y_test, probability, prediction, comparison, model, X_train, y_train)
    print(f'[{timing}] test ROC-AUC={metrics["test_roc_auc"]:.3f}, recall={metrics["test_recall"]:.3f}, F1={metrics["test_f1"]:.3f}', flush=True)
    return metrics


def write_results(root, results):
    rows = [
        '# Results', '',
        'Data: UCI Predict Students Dropout and Academic Success (4,424 rows).',
        'Outcome: Dropout vs Graduate, excluding unresolved Enrolled students.',
        '80/20 stratified train/test split, seed 42, five-fold cross-validation within training.',
        'All test results are for the selected model evaluated once on the held-out test set.', '',
        '| Prediction time | Chosen model | Features | Test ROC-AUC | Test recall | Test precision | Test F1 | Test accuracy |',
        '|---|---|---:|---:|---:|---:|---:|---:|'
    ]
    for r in results:
        rows.append(
            f'| {r["timing"].replace("_", " ")} | {r["chosen_model"]} | {r["feature_count"]} | '
            f'{r["test_roc_auc"]:.3f} | {r["test_recall"]:.3f} | {r["test_precision"]:.3f} | '
            f'{r["test_f1"]:.3f} | {r["test_accuracy"]:.3f} |'
        )
    rows += [
        '', 'The cutoff for predicting dropout is 0.50. Risk labels are display categories using',
        'illustrative 0.25 / 0.60 cutoffs, not proven thresholds for offering student support.',
        'This prototype is not suitable for real admissions, academic, or intervention decisions.',
        '', 'Read each experiment folder for the 4-model CV comparison, test plots, and predictions.'
    ]
    (root / 'RESULTS.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/data.csv')
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--timing', choices=['both'] + TIMINGS, default='both')
    args = parser.parse_args()
    df = read_dataset(args.data)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    selected = TIMINGS if args.timing == 'both' else [args.timing]
    results = [train_one(df, timing, root) for timing in selected]
    (root / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    write_results(root, results)
    print('Saved outputs to', root.resolve(), flush=True)


if __name__ == '__main__':
    main()
