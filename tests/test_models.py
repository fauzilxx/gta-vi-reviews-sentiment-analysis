"""Metric, data-isolation, serialization, and prediction-integrity regressions."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from src.artifacts import environment, sha256
from src.evaluate import run_evaluation, verify_evaluation
from src.modeling import CLASSES, compute_metrics, write_json
from src.train import run_training


class ModelTests(unittest.TestCase):
    def test_macro_and_weighted_metrics_with_unpredicted_minority(self):
        truth = ['Negative', 'Negative', 'Negative', 'Neutral', 'Positive']
        metrics, report, matrix = compute_metrics(truth, ['Negative'] * 5)
        self.assertAlmostEqual(metrics['accuracy'], .6)
        self.assertAlmostEqual(metrics['precision_macro'], .2)
        self.assertAlmostEqual(metrics['recall_macro'], 1 / 3)
        self.assertAlmostEqual(metrics['f1_macro'], .25)
        self.assertAlmostEqual(metrics['f1_weighted'], .45)
        self.assertEqual(report['Neutral']['f1-score'], 0)
        np.testing.assert_array_equal(matrix, [[3, 0, 0], [1, 0, 0], [1, 0, 0]])

    def test_training_isolation_reload_evaluation_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            features, models, reports = root / 'features', root / 'models', root / 'reports'
            features.mkdir()
            y_train = np.repeat(CLASSES, [12, 3, 6])
            X_train = sparse.csr_matrix(np.eye(3)[[CLASSES.index(y) for y in y_train]])
            y_test = np.array(CLASSES * 2)
            X_test = sparse.csr_matrix(np.vstack([np.eye(3), np.eye(3)[:2], np.zeros((1, 3))]))
            sparse.save_npz(features / 'X_train_tfidf.npz', X_train)
            sparse.save_npz(features / 'X_test_tfidf.npz', X_test)
            pd.DataFrame({'sentiment': y_train}).to_csv(features / 'y_train.csv', index=False)
            pd.DataFrame({'sentiment': y_test}).to_csv(features / 'y_test.csv', index=False)
            members = pd.DataFrame({'row_index': np.arange(27), 'review_id': [f'row-{i}' for i in range(27)],
                                    'split': ['train'] * 21 + ['test'] * 6,
                                    'matrix_row': list(range(21)) + list(range(6))})
            members.to_csv(features / 'split_membership.csv', index=False)
            write_json(features / 'manifest.json', {'environment': environment(),
                       'files': {path.as_posix(): sha256(path) for path in features.iterdir()}})
            loaded = []
            original_load = sparse.load_npz

            def record_load(filename):
                loaded.append(Path(filename).name)
                return original_load(filename)

            with contextlib.redirect_stdout(io.StringIO()), patch('src.train.sparse.load_npz', side_effect=record_load):
                training = run_training(features, models, reports)
            self.assertEqual(loaded, ['X_train_tfidf.npz'])
            self.assertTrue(all(item['reload_training_predictions_identical'] for item in training['models'].values()))
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_evaluation(features, models, reports)
            self.assertEqual(result['test_rows'], 6)
            self.assertEqual(result['zero_vector_test_rows'], 1)
            self.assertEqual(len(result['models']), 3)
            for item in result['models'].values():
                self.assertEqual(np.asarray(item['confusion_matrix']).sum(), 6)
            with self.assertRaises(FileExistsError):
                run_training(features, models, reports)
            with self.assertRaises(FileExistsError):
                run_evaluation(features, models, reports)
            # Mutate one saved report at a time; preserve the fixture after each case.
            mutations = [
                ('naive_bayes_classification_report.json',
                 lambda report: report['Negative'].__setitem__('precision', 0.123456), 'Classification report JSON'),
                ('naive_bayes_classification_report.json',
                 lambda report: report.pop('macro avg'), 'Classification report JSON'),
                ('naive_bayes_classification_report.json',
                 lambda report: report['macro avg'].__setitem__('f1-score', float('nan')), 'Classification report JSON'),
                ('evaluation.json',
                 lambda report: report['models']['linear_svm']['classification_report']['Neutral'].__setitem__('support', 999),
                 'Classification report gabungan'),
                ('evaluation.json', lambda report: report.__setitem__('test_rows', 999), 'Jumlah test'),
                ('evaluation.json', lambda report: report['models']['naive_bayes'].__setitem__('error_count', 999),
                 'Jumlah kesalahan'),
                ('evaluation.json',
                 lambda report: report['models']['naive_bayes']['subgroups']['zero_vector'].__setitem__('rows', 999),
                 'Metrik subgroup'),
                ('evaluation.json', lambda report: report['majority_baseline']['metrics'].__setitem__('accuracy', -1),
                 'Baseline mayoritas'),
                ('evaluation.json', lambda report: report['models'].__setitem__('fake_model', {}), 'Model/urutan kelas'),
            ]
            for filename, mutate, message in mutations:
                with self.subTest(file=filename, check=message):
                    path = reports / filename
                    original = path.read_bytes()
                    try:
                        changed = json.loads(original)
                        mutate(changed)
                        path.write_text(json.dumps(changed), encoding='utf-8')
                        with self.assertRaisesRegex(ValueError, message):
                            verify_evaluation(features, models, reports)
                    finally:
                        path.write_bytes(original)
            error_path = reports / 'misclassifications.csv'
            original_errors = error_path.read_bytes()
            changed_errors = pd.read_csv(error_path)
            changed_errors.loc[0, 'review_id'] = 'incorrect-row'
            changed_errors.to_csv(error_path, index=False)
            try:
                with self.assertRaisesRegex(ValueError, 'Daftar salah klasifikasi'):
                    verify_evaluation(features, models, reports)
            finally:
                error_path.write_bytes(original_errors)
            self.assertTrue(verify_evaluation(features, models, reports)['majority_baseline_matches'])
            saved = pd.read_csv(reports / 'predictions.csv')
            saved.loc[0, 'linear_svm'] = 'Neutral' if saved.loc[0, 'linear_svm'] != 'Neutral' else 'Positive'
            saved.to_csv(reports / 'predictions.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Prediksi tersimpan berbeda'):
                verify_evaluation(features, models, reports)


if __name__ == '__main__':
    unittest.main()
