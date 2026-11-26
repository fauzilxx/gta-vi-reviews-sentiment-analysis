"""Fixed Phase 4/5 configuration and shared metric definitions.

PRD.md specifies the algorithms, not hyperparameters. These baseline choices
are frozen before test predictions; no test-driven tuning is performed.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.artifacts import environment, sha256

CLASSES = ['Negative', 'Neutral', 'Positive']
MODEL_NAMES = {
    'naive_bayes': 'Multinomial Naive Bayes',
    'logistic_regression': 'Logistic Regression',
    'linear_svm': 'LinearSVC',
}


def make_models():
    return {
        'naive_bayes': MultinomialNB(alpha=1.0, fit_prior=True),
        'logistic_regression': LogisticRegression(
            C=1.0, class_weight='balanced', solver='lbfgs', max_iter=2000,
            tol=1e-4, random_state=42),
        'linear_svm': LinearSVC(
            C=1.0, class_weight='balanced', dual='auto', max_iter=10000,
            tol=1e-4, random_state=42),
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def verify_feature_files(features_dir):
    """Checksum and version checks only; no split, fit, or transformation."""
    manifest_path = Path(features_dir) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if environment() != manifest['environment']:
        raise RuntimeError('Versi environment berbeda dari pembuat artifact; periksa sebelum training/evaluasi')
    for filename, expected in manifest['files'].items():
        if sha256(filename) != expected:
            raise ValueError(f'Artifact Phase 3 berubah: {filename}')
    # Ensure requested paths are the same files validated by the manifest.
    validated = {Path(filename).resolve() for filename in manifest['files']}
    for filename in ['X_train_tfidf.npz', 'X_test_tfidf.npz', 'y_train.csv', 'y_test.csv', 'split_membership.csv']:
        if (Path(features_dir) / filename).resolve() not in validated:
            raise ValueError(f'Artifact tidak terdaftar pada manifest: {filename}')
    return manifest


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or not len(y_true):
        raise ValueError('Label dan prediksi harus berupa vektor nonkosong berukuran sama')
    if not set(y_true).issubset(CLASSES) or not set(y_pred).issubset(CLASSES):
        raise ValueError('Label/prediksi mengandung kelas yang tidak dikenal')
    report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(report['macro avg']['precision']),
        'recall_macro': float(report['macro avg']['recall']),
        'f1_macro': float(report['macro avg']['f1-score']),
        'precision_weighted': float(report['weighted avg']['precision']),
        'recall_weighted': float(report['weighted avg']['recall']),
        'f1_weighted': float(report['weighted avg']['f1-score']),
    }
    return metrics, report, confusion_matrix(y_true, y_pred, labels=CLASSES)
