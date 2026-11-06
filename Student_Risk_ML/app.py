from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from train import risk_level

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'

st.set_page_config(page_title='Student Support Risk Explorer', page_icon='🎓', layout='wide')
st.title('Student Support Risk Explorer')
st.caption('SFWRENG 4AL3 | Machine learning project | UCI dataset 697')
st.write('Explore predicted **dropout risk** and possible **support outreach priority**. These are research predictions, not verified academic-support needs.')

available = [x for x in ['enrollment', 'after_first_semester'] if (OUT / x / 'model.joblib').exists()]
if not available:
    st.error('No trained models found. Run: python train.py --timing both')
    st.stop()

with st.sidebar:
    st.header('Settings')
    timing = st.selectbox('Prediction timing', available, format_func=lambda x: 'At enrollment' if x == 'enrollment' else 'After first semester')
    st.write('Support priority thresholds')
    medium = st.slider('Moderate from', 0.05, 0.85, 0.25, 0.05)
    high = st.slider('High from', round(medium + 0.05, 2), 1.0, max(0.60, round(medium + 0.05, 2)), 0.05)
    st.caption('Illustrative thresholds, not validated intervention rules.')
    st.divider()
    st.caption('Dropout = 1, Graduate = 0. Unresolved Enrolled cases excluded from training.')

path = OUT / timing


@st.cache_resource
def get_model(model_file):
    return joblib.load(model_file)


@st.cache_data
def load_metrics(file):
    return pd.read_json(file, typ='series').to_dict()


artifact = get_model(str(path / 'model.joblib'))
metrics = load_metrics(str(path / 'metrics.json'))
cv = pd.read_csv(path / 'cross_validation.csv')

overview, comparisons, predictions, guide = st.tabs(['Overview', 'Model comparison', 'Try predictions', 'About the project'])

with overview:
    st.subheader('Held-out test results')
    columns = st.columns(4)
    for col, name, metric in zip(
        columns, ['ROC-AUC', 'Recall', 'Precision', 'F1 score'],
        ['test_roc_auc', 'test_recall', 'test_precision', 'test_f1']
    ):
        col.metric(name, f'{metrics[metric]:.1%}')
    st.caption(f'Chosen model: {metrics["chosen_model"].replace("_", " ").title()} · {metrics["test_rows"]} test records · {metrics["feature_count"]} predictors · 0.50 decision cutoff')
    st.write('**Recall** measures the share of actual dropout cases flagged by the model. **Precision** measures how often a dropout flag is correct.')
    left, right = st.columns(2)
    with left:
        st.image(str(path / 'confusion_matrix.png'), use_container_width=True)
        st.image(str(path / 'precision_recall.png'), use_container_width=True)
    with right:
        st.image(str(path / 'roc_curve.png'), use_container_width=True)
        st.image(str(path / 'feature_importance.png'), use_container_width=True)
    st.caption('Feature importance is measured by shuffling individual input columns on a sample of training data. It does not establish causation.')

with comparisons:
    st.subheader('Four algorithms, same training data')
    st.write('Five-fold stratified cross-validation happens only within the training split. The best average ROC-AUC selects the model, then it is tested on untouched data.')
    display_cols = ['model', 'cv_roc_auc', 'cv_average_precision', 'cv_recall', 'cv_precision', 'cv_f1', 'cv_accuracy']
    st.dataframe(cv[display_cols].round(3), hide_index=True, use_container_width=True)
    st.image(str(path / 'model_comparison.png'), use_container_width=True)
    st.caption('CV numbers are averages across five training folds, not test performance. Models may prioritize recall and precision differently.')
    st.download_button('Download cross-validation table', cv.to_csv(index=False), 'cross_validation.csv', 'text/csv')
    summary = OUT / 'summary.json'
    if summary.exists():
        st.subheader('Does first-semester data help?')
        rows = pd.read_json(summary)
        if len(rows) == 2:
            vals = rows[['timing', 'chosen_model', 'feature_count', 'test_roc_auc', 'test_recall', 'test_f1']].copy()
            st.dataframe(vals.round(3), hide_index=True, use_container_width=True)
            st.caption('This is a comparison on the same fixed split. It is not a guarantee of future performance.')

with predictions:
    st.subheader('Predict dropout risk for sample students')
    mode = st.radio('Input source', ['Use included sample', 'Upload CSV'], horizontal=True)
    if mode == 'Use included sample':
        st.caption('These are anonymized feature rows sampled from the test split. Edit any cell to try a hypothetical student.')
        raw = pd.read_csv(path / 'demo_students.csv')
        raw = st.data_editor(raw, use_container_width=True, hide_index=True, num_rows='fixed', key=f'edit_{timing}')
    else:
        uploaded = st.file_uploader('Upload student features as CSV', type='csv')
        if uploaded is None:
            st.info('Use the included sample to see predictions immediately, or upload a CSV that has the same columns as the training dataset.')
            st.stop()
        raw = pd.read_csv(uploaded, sep=None, engine='python')
        raw.columns = raw.columns.str.strip()
    missing = [x for x in artifact['features'] if x not in raw.columns]
    if missing:
        st.error('Missing fields: ' + ', '.join(missing))
        st.stop()
    X = raw[artifact['features']].apply(pd.to_numeric, errors='coerce')
    if X.empty:
        st.warning('No students found in this file.')
        st.stop()
    if X.isna().any().any():
        st.warning('Some cells are blank or nonnumeric. The model will fill missing values using training data.')
    probs = artifact['pipeline'].predict_proba(X)[:, 1]
    result = pd.DataFrame({
        'student_row': range(1, len(X) + 1),
        'dropout_probability': probs.round(4),
        'dropout_prediction': ['Dropout' if p >= 0.50 else 'Graduate' for p in probs],
        'support_priority': [risk_level(float(p), medium, high) for p in probs]
    })
    st.dataframe(result, hide_index=True, use_container_width=True)
    total_high = int((result['support_priority'] == 'High').sum())
    total_medium = int((result['support_priority'] == 'Moderate').sum())
    total_low = int((result['support_priority'] == 'Low').sum())
    c1, c2, c3 = st.columns(3)
    c1.metric('High priority', total_high)
    c2.metric('Moderate priority', total_medium)
    c3.metric('Low priority', total_low)
    st.bar_chart(result.set_index('student_row')['dropout_probability'], y_label='Dropout probability')
    st.download_button('Download predictions CSV', result.to_csv(index=False), 'student_predictions.csv', 'text/csv')
    st.caption('Model probabilities are not validated or calibrated for live student outreach. Support priority is illustrative.')

with guide:
    st.subheader('Research question')
    st.write('Can machine learning identify students at risk of dropping out early enough for academic support teams to consider proactive outreach?')
    st.subheader('Method')
    st.write('The dataset has 4,424 students, with 1,421 Dropout, 2,209 Graduate, and 794 Enrolled outcomes. Enrolled cases are excluded because their final outcome is unresolved. Four models are compared using stratified cross-validation on the training data. The top model is evaluated on a held-out 20% test split.')
    st.write('At enrollment, both semester grade groups are excluded. After first semester, second-semester grades are excluded. Coded categories are one-hot encoded, numeric predictors are scaled, and missing values are handled within the training pipeline.')
    st.subheader('Limits')
    st.write('This is an educational prototype from one higher-education context. Dropout risk is not a direct label of academic support need. Inputs are not individually time-stamped, so the enrollment feature set is an approximation of early prediction. Predictions could be biased, explanations are not causal, and other institutions or years have not been evaluated.')
    st.write('Outreach thresholds have not been validated. A real university intervention would need transparency, data protection, fairness auditing, calibration, and human review.')
    st.markdown('Dataset: [UCI Machine Learning Repository, dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), Realinho et al. (2021), CC BY 4.0.')

st.divider()
st.caption('Coursework demonstration only. Do not use this model to make consequential decisions about real students.')
