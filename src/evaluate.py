"""Phase 5: final evaluation of frozen models on saved test features.

Run from repository root: python -m src.evaluate
This module never fits a model/vectorizer or creates a split.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.artifacts import environment, sha256
from src.modeling import CLASSES, MODEL_NAMES, compute_metrics, verify_feature_files, write_json


def assert_report_matches(expected, saved, context):
    """Compare complete JSON reports, including aggregates and finite numeric values."""
    if isinstance(expected, dict):
        if not isinstance(saved, dict) or set(expected) != set(saved):
            raise ValueError(f'{context}: struktur laporan tidak sesuai')
        for key, value in expected.items():
            assert_report_matches(value, saved[key], f'{context}/{key}')
    elif isinstance(expected, list):
        if not isinstance(saved, list) or len(expected) != len(saved):
            raise ValueError(f'{context}: daftar tidak sesuai')
        for index, (value, actual) in enumerate(zip(expected, saved)):
            assert_report_matches(value, actual, f'{context}/{index}')
    elif isinstance(expected, (int, float, np.number)):
        if (isinstance(saved, bool) or not isinstance(saved, (int, float, np.number))
                or not np.isfinite(saved) or not np.isclose(expected, saved, rtol=0, atol=1e-12)):
            raise ValueError(f'{context}: nilai tidak sesuai')
    elif expected != saved:
        raise ValueError(f'{context}: nilai tidak sesuai')


def load_evaluation_inputs(features_dir, models_dir):
    features_dir, models_dir = Path(features_dir), Path(models_dir)
    verify_feature_files(features_dir)
    training = json.loads((models_dir / 'training_manifest.json').read_text(encoding='utf-8'))
    if training['environment'] != environment():
        raise ValueError('Environment model tidak sesuai')
    if training['feature_manifest_sha256'] != sha256(features_dir / 'manifest.json'):
        raise ValueError('Model dilatih dengan versi fitur berbeda')
    if sha256(training['plan_path']) != training['plan_sha256']:
        raise ValueError('Rencana model berubah sejak training')
    plan = json.loads(Path(training['plan_path']).read_text(encoding='utf-8'))
    if set(training['models']) != set(MODEL_NAMES):
        raise ValueError('Ketiga model belum lengkap')
    for name, item in training['models'].items():
        if (models_dir / f'{name}.pkl').resolve() != Path(item['path']).resolve():
            raise ValueError('Path model tidak sesuai manifest')
        if sha256(item['path']) != item['sha256']:
            raise ValueError(f'Model berubah: {name}')
        model = joblib.load(item['path'])
        if model.get_params() != item['parameters'] or model.get_params() != plan['models'][name]:
            raise ValueError(f'Parameter model berbeda dari rencana training: {name}')
        if list(model.classes_) != CLASSES:
            raise ValueError(f'Urutan kelas model tidak sesuai: {name}')
    X_test = sparse.load_npz(features_dir / 'X_test_tfidf.npz')
    y_test = pd.read_csv(features_dir / 'y_test.csv')['sentiment'].to_numpy()
    members = pd.read_csv(features_dir / 'split_membership.csv')
    members = members[members['split'] == 'test'].sort_values('matrix_row').reset_index(drop=True)
    if X_test.shape[0] != len(y_test) or len(members) != len(y_test):
        raise ValueError('Jumlah baris test/label/membership berbeda')
    if not np.array_equal(members['matrix_row'], np.arange(len(y_test))):
        raise ValueError('Urutan test membership tidak valid')
    return training, X_test, y_test, members


def verify_evaluation(features_dir, models_dir, reports_dir):
    """Independently reload models, regenerate predictions, and recalculate saved metrics."""
    reports_dir = Path(reports_dir)
    training, X_test, y_test, members = load_evaluation_inputs(features_dir, models_dir)
    predictions = pd.read_csv(reports_dir / 'predictions.csv')
    result = json.loads((reports_dir / 'evaluation.json').read_text(encoding='utf-8'))
    metrics_csv = pd.read_csv(reports_dir / 'metrics.csv').set_index('model')
    per_class = pd.read_csv(reports_dir / 'per_class_metrics.csv').set_index(['model', 'class'])
    if set(result['models']) != set(MODEL_NAMES) or result['classes'] != CLASSES:
        raise ValueError('Model/urutan kelas pada laporan tidak sesuai')
    if metrics_csv.index.has_duplicates or set(metrics_csv.index) != set(MODEL_NAMES):
        raise ValueError('Cakupan model pada metrik CSV tidak sesuai')
    expected_class_rows = {(name, label) for name in MODEL_NAMES for label in CLASSES}
    if per_class.index.has_duplicates or set(per_class.index) != expected_class_rows:
        raise ValueError('Cakupan model/kelas pada metrik per kelas tidak sesuai')
    assert_report_matches(len(y_test), result['test_rows'], 'Jumlah test')
    assert_report_matches(int((X_test.getnnz(axis=1) == 0).sum()), result['zero_vector_test_rows'], 'Jumlah vektor kosong')
    if result['selection_metric'] != 'f1_macro' or result['zero_division'] != 0:
        raise ValueError('Definisi evaluasi tidak sesuai')
    if result['training_manifest_sha256'] != sha256(Path(models_dir) / 'training_manifest.json'):
        raise ValueError('Laporan merujuk manifest training berbeda')
    if result['feature_manifest_sha256'] != sha256(Path(features_dir) / 'manifest.json'):
        raise ValueError('Laporan merujuk manifest fitur berbeda')
    for column in ['matrix_row', 'row_index', 'review_id']:
        if not np.array_equal(predictions[column].to_numpy(), members[column].to_numpy()):
            raise ValueError(f'Identitas prediksi tidak sesuai: {column}')
    if not np.array_equal(predictions.y_true.to_numpy(), y_test):
        raise ValueError('y_true tersimpan tidak sesuai artifact test')
    if not np.array_equal(predictions.zero_vector.to_numpy(), X_test.getnnz(axis=1) == 0):
        raise ValueError('Penanda vektor kosong tidak sesuai matriks')
    verified, expected_errors = {}, []
    for name, item in training['models'].items():
        model = joblib.load(item['path'])
        actual = model.predict(X_test)
        if not np.array_equal(actual, predictions[name].to_numpy()):
            raise ValueError(f'Prediksi tersimpan berbeda dari model yang dimuat ulang: {name}')
        metrics, report, matrix = compute_metrics(y_test, actual)
        saved_report = json.loads((reports_dir / f'{name}_classification_report.json').read_text(encoding='utf-8'))
        assert_report_matches(report, saved_report, f'Classification report JSON/{name}')
        assert_report_matches(report, result['models'][name]['classification_report'], f'Classification report gabungan/{name}')
        assert_report_matches(metrics, result['models'][name]['metrics'], f'Metrik JSON/{name}')
        for metric, value in metrics.items():
            if not np.isclose(value, result['models'][name]['metrics'][metric], rtol=0, atol=1e-12):
                raise ValueError(f'Metrik JSON tidak sesuai: {name}/{metric}')
            if not np.isclose(value, metrics_csv.loc[name, metric], rtol=0, atol=1e-12):
                raise ValueError(f'Metrik CSV tidak sesuai: {name}/{metric}')
        if not np.array_equal(matrix, result['models'][name]['confusion_matrix']):
            raise ValueError('Confusion matrix JSON tidak sesuai')
        saved_matrix = pd.read_csv(reports_dir / f'{name}_confusion_matrix.csv', index_col=0)
        if list(saved_matrix.index) != CLASSES or list(saved_matrix.columns) != CLASSES or not np.array_equal(matrix, saved_matrix.to_numpy()):
            raise ValueError('Confusion matrix CSV tidak sesuai')
        for label in CLASSES:
            for key in ['precision', 'recall', 'f1-score', 'support']:
                if not np.isclose(report[label][key], per_class.loc[(name, label), key], rtol=0, atol=1e-12):
                    raise ValueError('Metrik per kelas tidak sesuai')
        if int(matrix.sum()) != len(y_test):
            raise ValueError('Confusion matrix tidak mencakup seluruh test')
        if not np.allclose(matrix.sum(axis=1), [report[label]['support'] for label in CLASSES]):
            raise ValueError('Support kelas tidak cocok dengan confusion matrix')
        mismatch = actual != y_test
        assert_report_matches(int(mismatch.sum()), result['models'][name]['error_count'], f'Jumlah kesalahan/{name}')
        pairs = [{'actual': actual_label, 'predicted': predicted_label, 'count': int(matrix[i, j])}
                 for i, actual_label in enumerate(CLASSES) for j, predicted_label in enumerate(CLASSES)
                 if i != j and matrix[i, j] > 0]
        pairs.sort(key=lambda pair: -pair['count'])
        assert_report_matches(pairs, result['models'][name]['error_pairs'], f'Pasangan kesalahan/{name}')
        subsets = {}
        empty = X_test.getnnz(axis=1) == 0
        for subgroup, mask in [('zero_vector', empty), ('nonzero_vector', ~empty)]:
            subsets[subgroup] = {'rows': int(mask.sum())}
            if mask.any():
                subsets[subgroup]['metrics'] = compute_metrics(y_test[mask], actual[mask])[0]
        assert_report_matches(subsets, result['models'][name]['subgroups'], f'Metrik subgroup/{name}')
        errors = predictions.loc[mismatch, ['matrix_row', 'row_index', 'review_id', 'y_true', 'zero_vector']].copy()
        errors['model'], errors['y_pred'] = name, actual[mismatch]
        expected_errors.append(errors)
        verified[name] = {'reload_test_predictions_identical': True, 'metrics_recomputed_match': True,
                          'confusion_matrix_covers_all_test_rows': True, 'classification_reports_match': True,
                          'error_summaries_match': True, 'subgroup_metrics_match': True}
    saved_errors = pd.read_csv(reports_dir / 'misclassifications.csv')
    expected_errors = pd.concat(expected_errors, ignore_index=True)
    if list(saved_errors.columns) != list(expected_errors.columns):
        raise ValueError('Kolom daftar salah klasifikasi tidak sesuai')
    try:
        pd.testing.assert_frame_equal(
            expected_errors.sort_values(['model', 'matrix_row']).reset_index(drop=True),
            saved_errors.sort_values(['model', 'matrix_row']).reset_index(drop=True),
            check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError('Daftar salah klasifikasi tidak sesuai prediksi aktual') from exc
    majority_class = max(training['training_class_counts'], key=training['training_class_counts'].get)
    expected_baseline = {'class_selected_from_train': majority_class,
                         'metrics': compute_metrics(y_test, np.repeat(majority_class, len(y_test)))[0]}
    assert_report_matches(expected_baseline, result['majority_baseline'], 'Baseline mayoritas')
    ranked = sorted(result['models'], key=lambda name: (-result['models'][name]['metrics']['f1_macro'], name))
    if result['ranking_macro_f1'] != ranked:
        raise ValueError('Urutan model tidak sesuai Macro F1')
    return {'test_rows': len(y_test), 'models': verified, 'misclassifications_match': True,
            'majority_baseline_matches': True,
            'feature_manifest_unchanged': True, 'predictions_sha256': sha256(reports_dir / 'predictions.csv')}


def run_evaluation(features_dir='data/processed/grouped_v1', models_dir='models/grouped_v1',
                   reports_dir='reports/grouped_v1'):
    reports_dir = Path(reports_dir)
    if any((reports_dir / name).exists() for name in ['evaluation.json', 'predictions.csv', 'metrics.csv']):
        raise FileExistsError('Hasil evaluasi sudah ada; gunakan direktori hasil baru agar tidak tertimpa')
    training, X_test, y_test, members = load_evaluation_inputs(features_dir, models_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    empty = X_test.getnnz(axis=1) == 0
    predictions = members[['matrix_row', 'row_index', 'review_id']].copy()
    predictions['y_true'], predictions['zero_vector'] = y_test, empty
    all_results, summary_rows, class_rows, error_rows = {}, [], [], []
    for name, item in training['models'].items():
        model = joblib.load(item['path'])
        predicted = model.predict(X_test)
        predictions[name] = predicted
        metrics, report, matrix = compute_metrics(y_test, predicted)
        summary_rows.append({'model': name, **metrics})
        class_rows.extend({'model': name, 'class': label, **report[label]} for label in CLASSES)
        confusion = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
        confusion.index.name = 'actual'
        confusion.to_csv(reports_dir / f'{name}_confusion_matrix.csv')
        write_json(reports_dir / f'{name}_classification_report.json', report)
        errors = predictions.loc[predicted != y_test, ['matrix_row', 'row_index', 'review_id', 'y_true', 'zero_vector']].copy()
        errors['model'], errors['y_pred'] = name, predicted[predicted != y_test]
        error_rows.append(errors)
        subset = {}
        for label, mask in [('zero_vector', empty), ('nonzero_vector', ~empty)]:
            subset[label] = {'rows': int(mask.sum())}
            if mask.any():
                subset[label]['metrics'] = compute_metrics(y_test[mask], predicted[mask])[0]
        pairs = [{'actual': actual, 'predicted': pred, 'count': int(matrix[i, j])}
                 for i, actual in enumerate(CLASSES) for j, pred in enumerate(CLASSES)
                 if i != j and matrix[i, j] > 0]
        pairs.sort(key=lambda pair: -pair['count'])
        all_results[name] = {'display_name': MODEL_NAMES[name], 'metrics': metrics,
                             'classification_report': report, 'confusion_matrix': matrix.tolist(),
                             'error_count': int((predicted != y_test).sum()),
                             'error_pairs': pairs, 'subgroups': subset}
        print(f'{name}: Macro F1={metrics["f1_macro"]:.6f}; Accuracy={metrics["accuracy"]:.6f}', flush=True)
    predictions.to_csv(reports_dir / 'predictions.csv', index=False)
    pd.DataFrame(summary_rows).sort_values('f1_macro', ascending=False).to_csv(reports_dir / 'metrics.csv', index=False)
    pd.DataFrame(class_rows).to_csv(reports_dir / 'per_class_metrics.csv', index=False)
    pd.concat(error_rows, ignore_index=True).to_csv(reports_dir / 'misclassifications.csv', index=False)
    majority_class = max(training['training_class_counts'], key=training['training_class_counts'].get)
    baseline_metrics = compute_metrics(y_test, np.repeat(majority_class, len(y_test)))[0]
    result = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'selection_metric': 'f1_macro', 'averaging': 'Precision/Recall in main table are macro averages.',
        'zero_division': 0,
        'classes': CLASSES, 'test_rows': len(y_test), 'zero_vector_test_rows': int(empty.sum()),
        'models': all_results, 'majority_baseline': {'class_selected_from_train': majority_class, 'metrics': baseline_metrics},
        'ranking_macro_f1': sorted(all_results, key=lambda name: (-all_results[name]['metrics']['f1_macro'], name)),
        'training_manifest_sha256': sha256(Path(models_dir) / 'training_manifest.json'),
        'feature_manifest_sha256': sha256(Path(features_dir) / 'manifest.json'),
        'source_hashes': {name: sha256(name) for name in ['src/evaluate.py', 'src/modeling.py']},
        'limitations': ['Satu held-out test; peringkat bukan bukti signifikansi statistik.',
                        'Tidak ada tuning setelah melihat test; test tidak lagi boleh dipakai untuk tuning berikutnya.',
                        'Label merupakan proksi rating dan bukan anotasi manusia.',
                        'Subset deduplikasi ID dan cleaning legacy diwarisi dari eksperimen sebelumnya.',
                        'Vektor kosong tetap dihitung; subgroup kecil tidak mewakili seluruh dataset.',
                        'Versi/lisensi Kaggle dan semantik review_id belum terverifikasi independen.'],
    }
    write_json(reports_dir / 'evaluation.json', result)
    make_figures(result, reports_dir)
    write_report(result, reports_dir)
    verification = verify_evaluation(features_dir, models_dir, reports_dir)
    verification['files'] = {path.as_posix(): sha256(path) for path in reports_dir.rglob('*')
                              if path.is_file() and path.name != 'evaluation_verification.json'}
    write_json(reports_dir / 'evaluation_verification.json', verification)
    return result


def make_figures(result, reports_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    folder = reports_dir / 'figures'
    folder.mkdir(exist_ok=True)
    names = list(MODEL_NAMES)
    figure, ax = plt.subplots(figsize=(11, 5))
    metrics = [('accuracy', 'Accuracy'), ('precision_macro', 'Precision macro'), ('recall_macro', 'Recall macro'),
               ('f1_macro', 'Macro F1'), ('f1_weighted', 'Weighted F1')]
    x, width = np.arange(len(metrics)), .24
    for index, name in enumerate(names):
        ax.bar(x + (index - 1) * width, [result['models'][name]['metrics'][key] for key, _ in metrics],
               width=width, label=MODEL_NAMES[name])
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title(f'Final held-out evaluation ({result["test_rows"]:,} reviews)')
    ax.legend(loc='lower right')
    figure.tight_layout()
    figure.savefig(folder / 'model_comparison.png', dpi=160)
    plt.close(figure)
    for normalized in [False, True]:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        for name, ax in zip(names, axes):
            matrix = np.array(result['models'][name]['confusion_matrix'])
            if normalized:
                matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                                   out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
            ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(
                ax=ax, colorbar=False, cmap='Blues', values_format='.2f' if normalized else 'd')
            ax.set_title(MODEL_NAMES[name])
        figure.suptitle('Confusion matrices: row-normalized' if normalized else 'Confusion matrices: counts')
        figure.tight_layout()
        figure.savefig(folder / ('confusion_matrices_normalized.png' if normalized else 'confusion_matrices.png'), dpi=160)
        plt.close(figure)
    figure, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASSES))
    for index, name in enumerate(names):
        ax.bar(x + (index - 1) * .24,
               [result['models'][name]['classification_report'][label]['f1-score'] for label in CLASSES],
               width=.24, label=MODEL_NAMES[name])
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1-score')
    ax.set_title('Per-class F1: minority Neutral retained in evaluation')
    ax.legend(loc='lower right')
    figure.tight_layout()
    figure.savefig(folder / 'per_class_f1.png', dpi=160)
    plt.close(figure)


def write_report(result, reports_dir):
    lines = ['# Hasil Phase 4–5: klasifikasi sentimen GTA VI', '',
             f'Evaluasi aktual terhadap {result["test_rows"]:,} ulasan test dari artifact grouped_v1. '
             'Split dan TF-IDF tidak dibuat ulang. Tidak ada tuning atau pemilihan konfigurasi menggunakan test.', '',
             'Konfigurasi dibekukan di model_plan.json sebelum training. PRD.md menetapkan algoritma, '
             'sedangkan hyperparameter baseline merupakan keputusan implementasi. LR/SVM menggunakan '
             'class_weight=balanced; Naive Bayes memakai prior kelas empiris training.', '',
             'Precision dan Recall pada tabel utama memakai macro average. Seluruh skor ditampilkan dalam persen. '
             'Metrik dengan pembagi nol diberi nilai 0 (zero_division=0).', '',
             '| Model | Accuracy | Precision macro | Recall macro | Macro F1 | Weighted F1 |',
             '|---|---:|---:|---:|---:|---:|']
    for name in result['ranking_macro_f1']:
        metrics = result['models'][name]['metrics']
        scores = [100 * metrics[key] for key in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']]
        lines.append('| ' + MODEL_NAMES[name] + ' | ' + ' | '.join(f'{score:.2f}' for score in scores) + ' |')
    lines += ['', '| Model | Kelas | Precision | Recall | F1 | Support |', '|---|---|---:|---:|---:|---:|']
    for name in result['ranking_macro_f1']:
        for label in CLASSES:
            scores = result['models'][name]['classification_report'][label]
            lines.append(f'| {MODEL_NAMES[name]} | {label} | {scores["precision"]*100:.2f} | '
                         f'{scores["recall"]*100:.2f} | {scores["f1-score"]*100:.2f} | {int(scores["support"])} |')
    best = result['ranking_macro_f1'][0]
    lines += ['', f'Peringkat pertama berdasarkan Macro F1 pada test ini: **{MODEL_NAMES[best]}**. '
              'Peringkat ini tidak membuktikan signifikansi statistik dan tidak digunakan untuk tuning ulang.', '',
              'Baseline kelas mayoritas dipilih dari training: '
              f'{result["majority_baseline"]["class_selected_from_train"]}. '
              f'Accuracy={result["majority_baseline"]["metrics"]["accuracy"]*100:.2f}%; '
              f'Macro F1={result["majority_baseline"]["metrics"]["f1_macro"]*100:.2f}%.', '',
              f'Seluruh {result["zero_vector_test_rows"]} vektor test kosong tetap masuk evaluasi. '
              'Rincian subgroup tersedia pada evaluation.json.', '',
              '## Analisis kesalahan', '']
    for name in result['ranking_macro_f1']:
        item = result['models'][name]
        pairs = '; '.join(f'{pair["actual"]} → {pair["predicted"]}: {pair["count"]}' for pair in item['error_pairs'][:3])
        lines.append(f'- {MODEL_NAMES[name]}: {item["error_count"]} kesalahan. Pasangan terbanyak: {pairs}.')
    lines += ['', 'Kesalahan di atas merupakan pola kebingungan label, bukan diagnosis penyebab bahasa/sarkasme '
              'yang belum dianotasi. Identitas setiap kesalahan tersedia di misclassifications.csv.', '',
              '## Artifact dan verifikasi', '',
              'Model Joblib dan training_manifest.json berada di models/grouped_v1. Prediksi untuk setiap baris test '
              'ada di predictions.csv. Classification report, confusion matrix CSV, grafik, dan metrik agregat '
              'tersimpan pada direktori laporan ini. evaluation_verification.json mencatat pemuatan ulang, '
              'prediksi ulang, dan perhitungan ulang metrik dari prediksi aktual.', '',
              '![Perbandingan model](figures/model_comparison.png)', '',
              '![Confusion matrix](figures/confusion_matrices.png)', '',
              '![F1 per kelas](figures/per_class_f1.png)', '', '## Keterbatasan', '']
    lines.extend('- ' + note for note in result['limitations'])
    (reports_dir / 'evaluation_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', default='data/processed/grouped_v1')
    parser.add_argument('--models-dir', default='models/grouped_v1')
    parser.add_argument('--reports-dir', default='reports/grouped_v1')
    parser.add_argument('--verify-only', action='store_true', help='Periksa hasil tersimpan tanpa menulis atau melatih ulang')
    args = parser.parse_args()
    arguments = vars(args)
    verify_only = arguments.pop('verify_only')
    if verify_only:
        print(json.dumps(verify_evaluation(**arguments), indent=2))
    else:
        run_evaluation(**arguments)
