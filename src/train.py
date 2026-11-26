"""Phase 4: train fixed classifiers on saved TF-IDF training data only.

From repository root: python -m src.train
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning

from src.artifacts import environment, sha256
from src.modeling import CLASSES, make_models, verify_feature_files, write_json


def run_training(features_dir='data/processed/grouped_v1', models_dir='models/grouped_v1',
                 reports_dir='reports/grouped_v1'):
    features_dir, models_dir, reports_dir = map(Path, [features_dir, models_dir, reports_dir])
    models = make_models()
    targets = [models_dir / f'{name}.pkl' for name in models]
    targets += [models_dir / 'training_manifest.json', reports_dir / 'model_plan.json']
    if any(path.exists() for path in targets):
        raise FileExistsError('Output training sudah ada; gunakan versi direktori baru agar hasil tidak tertimpa')
    feature_manifest = verify_feature_files(features_dir)
    X_train = sparse.load_npz(features_dir / 'X_train_tfidf.npz')
    y_train = pd.read_csv(features_dir / 'y_train.csv')['sentiment']
    if X_train.shape[0] != len(y_train) or set(y_train) != set(CLASSES):
        raise ValueError('Ukuran/kelas data training tidak valid')
    if not np.isfinite(X_train.data).all() or (X_train.data < 0).any():
        raise ValueError('Fitur harus finite dan nonnegative untuk MultinomialNB')
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'instructions': 'PRD.md',
        'configuration_basis': 'Fixed baseline chosen before test predictions; PRD contains no numerical hyperparameters.',
        'selection_metric': 'f1_macro', 'test_usage': 'Final evaluation only; no tuning or early stopping.',
        'imbalance': 'LR/SVM balanced weights computed from training labels; NB uses empirical training class priors.',
        'models': {name: estimator.get_params() for name, estimator in models.items()},
        'classes': CLASSES, 'training_shape': list(X_train.shape),
        'training_class_counts': y_train.value_counts().to_dict(),
        'feature_manifest_sha256': sha256(features_dir / 'manifest.json'),
        'phase3_input_hashes': feature_manifest['files'], 'environment': environment(),
    }
    write_json(reports_dir / 'model_plan.json', plan)
    results = {}
    for name, estimator in models.items():
        print(f'Training {name} ...', flush=True)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            estimator.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        if any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise RuntimeError(f'{name} belum konvergen; training dihentikan tanpa memakai test untuk tuning')
        if set(estimator.classes_) != set(CLASSES) or estimator.n_features_in_ != X_train.shape[1]:
            raise ValueError(f'{name}: kelas/dimensi hasil training tidak sesuai')
        filename = models_dir / f'{name}.pkl'
        joblib.dump(estimator, filename)
        reloaded = joblib.load(filename)
        same = np.array_equal(estimator.predict(X_train), reloaded.predict(X_train))
        if not same:
            raise ValueError(f'Prediksi {name} berubah setelah serialisasi')
        iterations = getattr(estimator, 'n_iter_', None)
        results[name] = {
            'path': filename.as_posix(), 'sha256': sha256(filename),
            'estimator': type(estimator).__name__, 'parameters': estimator.get_params(),
            'training_seconds': elapsed,
            'n_iter': np.asarray(iterations).tolist() if iterations is not None else None,
            'warnings': [str(item.message) for item in captured],
            'reload_training_predictions_identical': bool(same),
        }
        print(f'{name}: saved; reload cocok; {elapsed:.2f} detik', flush=True)
    # Also prove that reading/training did not mutate the Phase 3 artifacts.
    verify_feature_files(features_dir)
    manifest = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'features_dir': features_dir.as_posix(), 'classes': CLASSES,
        'feature_manifest_sha256': plan['feature_manifest_sha256'],
        'plan_path': (reports_dir / 'model_plan.json').as_posix(),
        'plan_sha256': sha256(reports_dir / 'model_plan.json'),
        'training_class_counts': plan['training_class_counts'],
        'environment': environment(), 'models': results,
        'source_hashes': {name: sha256(name) for name in ['src/train.py', 'src/modeling.py']},
    }
    write_json(models_dir / 'training_manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', default='data/processed/grouped_v1')
    parser.add_argument('--models-dir', default='models/grouped_v1')
    parser.add_argument('--reports-dir', default='reports/grouped_v1')
    args = parser.parse_args()
    run_training(**vars(args))
