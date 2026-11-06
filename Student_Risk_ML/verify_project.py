import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from streamlit.testing.v1 import AppTest

root = Path.cwd()
report = {'python': sys.version, 'checks': {}, 'models': {}}
for name, args in [('dependencies', ['-m', 'pip', 'check']), ('unit_tests', ['-m', 'unittest', 'test_pipeline.py', '-v'])]:
    run = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    report['checks'][name] = {'passed': run.returncode == 0, 'output': run.stdout + run.stderr}
    assert run.returncode == 0, report['checks'][name]
for timing in ['enrollment', 'after_first_semester']:
    folder = root / 'outputs' / timing
    artifact = joblib.load(folder / 'model.joblib')
    rows = pd.read_csv(folder / 'test_predictions.csv')
    probability = artifact['pipeline'].predict_proba(rows[artifact['features']])[:, 1]
    np.testing.assert_allclose(probability, rows['dropout_probability'], rtol=1e-6, atol=1e-7)
    metrics = json.loads((folder / 'metrics.json').read_text())
    auc = roc_auc_score(rows.actual_dropout, probability)
    accuracy = accuracy_score(rows.actual_dropout, probability >= .5)
    assert abs(auc - metrics['test_roc_auc']) < 1e-10
    assert abs(accuracy - metrics['test_accuracy']) < 1e-10
    report['models'][timing] = {'passed': True, 'test_rows': len(rows), 'roc_auc': auc, 'accuracy': accuracy, 'model': artifact['model_name']}
app = AppTest.from_file(str(root / 'app.py'), default_timeout=60).run()
assert not app.exception, str(app.exception)
app.selectbox[0].select('after_first_semester').run()
assert not app.exception, str(app.exception)
report['checks']['dashboard_execution'] = {'passed': True, 'detail': 'Both prediction timings rendered and executed sample inference with Streamlit AppTest.'}
with (root / 'outputs' / 'streamlit_launch.log').open('w') as log:
    server = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.headless=true', '--server.address=127.0.0.1', '--server.port=8507', '--browser.gatherUsageStats=false'], stdout=log, stderr=subprocess.STDOUT)
    try:
        for attempt in range(60):
            try:
                with urllib.request.urlopen('http://127.0.0.1:8507/_stcore/health', timeout=2) as response:
                    status = response.status
                    body = response.read().decode()
                break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError('Streamlit exited; see launch log')
                time.sleep(1)
        else:
            raise RuntimeError('Streamlit health check timed out')
        assert status == 200 and body == 'ok'
        with urllib.request.urlopen('http://127.0.0.1:8507/', timeout=5) as response:
            assert response.status == 200
        report['checks']['streamlit_server'] = {'passed': True, 'health_status': status, 'health_body': body, 'homepage_status': 200, 'port': 8507, 'stopped_after_check': True}
    finally:
        server.terminate()
        server.wait(timeout=15)
report['dataset_sha256'] = hashlib.sha256((root / 'data/data.csv').read_bytes()).hexdigest()
(root / 'outputs/verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
