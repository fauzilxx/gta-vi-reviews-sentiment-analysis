"""Exercise an already running local Streamlit Agent API; never train or fit.

Start dashboard/app.py with --server.enableAgentApi=true on port 8765, then run
python scripts/check_dashboard_live.py. This API flag is for local testing only.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.modeling import MODEL_NAMES
from src.predict import PROJECT_ROOT, load_model_bundle, predict_sentiment

BASE = 'http://127.0.0.1:8765'


def interact(**payload):
    request = urllib.request.Request(BASE + '/_stcore/agent/interact',
                                     data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        snapshot = json.load(response)
    if snapshot.get('status') != 'ready':
        raise AssertionError(snapshot)
    if any(node['type'] in ['error', 'exception'] for node in nodes(snapshot['tree'])):
        raise AssertionError(snapshot['tree'])
    return snapshot


def nodes(node):
    yield node
    for child in node.get('children', []):
        yield from nodes(child)


def run():
    # The protocol document was read in this session before driving any widgets.
    bundle = load_model_bundle()
    snapshot = interact()
    session = snapshot['session_id']
    checks = []
    observed_metrics = {}
    for page in ['', 'eda', 'comparison', 'evaluation', 'prediction']:
        snapshot = interact(session_id=session, page=page)
        checks.append({'page': page or 'overview', 'status': snapshot['status']})
        elements = list(nodes(snapshot['tree']))
        if page == 'comparison':
            table = next(node['data'] for node in elements if node['type'] == 'dataframe')
            columns = [column['name'] for column in table['columns']]
            for row in table['preview']['rows']:
                record = dict(zip(columns, row))
                name = next(key for key, value in MODEL_NAMES.items() if value == record['Model'])
                for column, metric in [('Accuracy', 'accuracy'), ('Precision macro', 'precision_macro'),
                                       ('Recall macro', 'recall_macro'), ('Macro F1', 'f1_macro'), ('Weighted F1', 'f1_weighted')]:
                    assert abs(record[column] - bundle.evaluation['models'][name]['metrics'][metric] * 100) < 1e-10
                observed_metrics[name] = record
        if page == 'eda':
            snapshot = interact(session_id=session, widget_state={'word_limit': 30})
            charts = [node for node in nodes(snapshot['tree']) if node['type'] == 'altair_chart']
            assert len(charts) == 5 and charts[-1]['data']['row_count'] == 30
            checks.append({'interaction': '30 frequent words', 'status': 'passed'})
        if page == 'evaluation':
            for name, label in MODEL_NAMES.items():
                snapshot = interact(session_id=session, widget_state={'evaluation_model': label})
                metrics = {node['props']['label']: node['props']['value'] for node in nodes(snapshot['tree']) if node['type'] == 'metric'}
                assert metrics['Macro F1'] == f'{bundle.evaluation["models"][name]["metrics"]["f1_macro"]:.2%}'
            snapshot = interact(session_id=session, widget_state={'normalized_matrix': True})
            chart = next(node['data'] for node in nodes(snapshot['tree']) if node['type'] == 'altair_chart')
            assert chart['spec']['layer'][0]['encoding']['color']['scale']['domain'] == [0, 1]
            checks.append({'interaction': 'all evaluation models and normalized matrix', 'status': 'passed'})
    text = 'I do not like this boring game and its terrible performance.'
    for name, label in MODEL_NAMES.items():
        snapshot = interact(session_id=session, widget_state={'prediction_model': label, 'review_text': text}, trigger={'key': 'predict_submit'})
        expected = predict_sentiment(text, name, bundle)['label']
        headings = [node['props']['body'] for node in nodes(snapshot['tree']) if node['type'] == 'subheader']
        assert expected in headings
        checks.append({'interaction': 'prediction', 'model': name, 'expected': expected, 'status': 'passed'})
    for text in ['', '12345 !!!', 'qzxvnonexistenttokenxx qzxvnonexistenttokenyy']:
        snapshot = interact(session_id=session, widget_state={'review_text': text}, trigger={'key': 'predict_submit'})
        assert any(node['type'] == 'warning' for node in nodes(snapshot['tree']))
        checks.append({'interaction': 'invalid input rejected', 'input': text, 'status': 'passed'})
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'base_url': BASE,
              'checks': checks, 'dashboard_metrics_match_experiment': True,
              'observed_comparison_table': observed_metrics}
    destination = PROJECT_ROOT / 'reports/dashboard_live_checks.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
