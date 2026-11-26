"""Read-only model persistence and inference, independent of Streamlit and cwd."""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.artifacts import environment, sha256
from src.modeling import CLASSES, MODEL_NAMES
from src.preprocessing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelBundle:
    models: dict[str, Any]
    vectorizer: Any
    feature_manifest: dict
    training_manifest: dict
    evaluation: dict


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve_project_file(root, filename):
    root = Path(root).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path artifact berada di luar direktori proyek')
    return path


def artifact_signature(root=PROJECT_ROOT):
    """Cheap cache invalidation when a file used by the dashboard changes."""
    root = Path(root)
    files = [root / 'data/processed/processed_reviews.csv', root / 'src/preprocessing.py',
             *sorted((root / 'data/processed/grouped_v1').glob('*')),
             *sorted((root / 'models/grouped_v1').glob('*')),
             *sorted((root / 'reports/grouped_v1').rglob('*'))]
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                 for path in files if path.is_file())


def load_model_bundle(root=PROJECT_ROOT):
    """Validate saved artifact checksums/versions and load; never fit or split."""
    root = Path(root).resolve()
    features_path = root / 'data/processed/grouped_v1/manifest.json'
    training_path = root / 'models/grouped_v1/training_manifest.json'
    features = read_json(features_path)
    training = read_json(training_path)
    evaluation = read_json(root / 'reports/grouped_v1/evaluation.json')
    verification = read_json(root / 'reports/grouped_v1/evaluation_verification.json')
    if environment() != features['environment'] or environment() != training['environment']:
        raise ValueError('Versi Python/paket tidak sesuai environment model tersimpan')
    if sha256(features_path) != training['feature_manifest_sha256'] or sha256(features_path) != evaluation['feature_manifest_sha256']:
        raise ValueError('Manifest fitur tidak sesuai eksperimen')
    if sha256(training_path) != evaluation['training_manifest_sha256']:
        raise ValueError('Manifest model tidak sesuai hasil evaluasi')
    for paths in [features['files'], verification['files']]:
        for filename, expected in paths.items():
            if sha256(resolve_project_file(root, filename)) != expected:
                raise ValueError(f'Checksum artifact tidak sesuai: {Path(filename).name}')
    if sha256(root / 'data/processed/processed_reviews.csv') != features['dataset_sha256']:
        raise ValueError('Dataset processed berbeda dari eksperimen')
    if features['cleaning_version'] != 'legacy-v1':
        raise ValueError('Versi cleaning artifact belum didukung untuk eksperimen ini')
    if sha256(root / 'src/preprocessing.py') != features['source_files']['src/preprocessing.py']:
        raise ValueError('Kode preprocessing berbeda dari versi yang membentuk dataset training')
    plan_path = resolve_project_file(root, training['plan_path'])
    if sha256(plan_path) != training['plan_sha256']:
        raise ValueError('Rencana training tidak sesuai model tersimpan')
    plan = read_json(plan_path)
    if set(training['models']) != set(MODEL_NAMES) or set(evaluation['models']) != set(MODEL_NAMES):
        raise ValueError('Ketiga model belum lengkap')
    vector_path = root / 'models/grouped_v1/tfidf_vectorizer.pkl'
    if vector_path.resolve() not in {resolve_project_file(root, name) for name in features['files']}:
        raise ValueError('Vectorizer tidak terdaftar pada manifest')
    vectorizer = joblib.load(vector_path)
    models = {}
    for name, info in training['models'].items():
        path = resolve_project_file(root, info['path'])
        if sha256(path) != info['sha256']:
            raise ValueError(f'Checksum model tidak sesuai: {name}')
        model = joblib.load(path)
        if model.n_features_in_ != len(vectorizer.vocabulary_) or list(model.classes_) != CLASSES:
            raise ValueError(f'Dimensi/kelas model tidak sesuai vectorizer: {name}')
        if model.get_params() != info['parameters'] or model.get_params() != plan['models'][name]:
            raise ValueError(f'Konfigurasi model tidak sesuai rencana: {name}')
        models[name] = model
    if evaluation['classes'] != CLASSES:
        raise ValueError('Urutan kelas evaluasi tidak sesuai')
    ranking = sorted(evaluation['models'], key=lambda name: (-evaluation['models'][name]['metrics']['f1_macro'], name))
    if evaluation['ranking_macro_f1'] != ranking:
        raise ValueError('Peringkat Macro F1 tidak sesuai metrik')
    return ModelBundle(models, vectorizer, features, training, evaluation)


def predict_sentiment(text, model_name, bundle):
    if model_name not in bundle.models:
        raise ValueError('Algoritma tidak dikenal')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Tuliskan ulasan terlebih dahulu.')
    cleaned = clean_text(text, version=bundle.feature_manifest['cleaning_version'])
    if not cleaned:
        raise ValueError('Ulasan tidak memiliki teks yang dapat dianalisis setelah pembersihan.')
    matrix = bundle.vectorizer.transform([cleaned])
    if matrix.nnz == 0:
        raise ValueError('Tidak ada kata atau frasa yang dikenali model. Coba ulasan yang lebih lengkap.')
    label = str(bundle.models[model_name].predict(matrix)[0])
    if label not in CLASSES:
        raise ValueError('Model menghasilkan kelas yang tidak dikenal')
    return {'label': label, 'model': model_name, 'clean_text': cleaned, 'recognized_features': int(matrix.nnz)}


def load_reviews(root=PROJECT_ROOT):
    root = Path(root)
    path = root / 'data/processed/processed_reviews.csv'
    manifest = read_json(root / 'data/processed/grouped_v1/manifest.json')
    if sha256(path) != manifest['dataset_sha256']:
        raise ValueError('Dataset berubah dari eksperimen')
    frame = pd.read_csv(path, usecols=['platform', 'published_date', 'score', 'clean_text', 'sentiment'])
    if frame.isna().any().any() or set(frame.sentiment) != set(CLASSES):
        raise ValueError('Dataset tidak lengkap')
    expected = np.select([frame.score <= 4, frame.score <= 7], ['Negative', 'Neutral'], default='Positive')
    if not frame.score.between(0, 10).all() or not frame.sentiment.eq(expected).all():
        raise ValueError('Label dataset tidak sesuai strategi rating')
    frame['text_length'] = frame.clean_text.str.len()
    return frame


def eda_summary(frame, vectorizer):
    """Descriptive counts only, with no fitting or changes to model vocabulary."""
    counts = Counter()
    tokenize = vectorizer.build_tokenizer()
    for text in frame.clean_text:
        counts.update(tokenize(text))
    lengths, edges = np.histogram(frame.text_length, bins=40)
    return {
        'words': pd.DataFrame(counts.most_common(30), columns=['Kata', 'Kemunculan']),
        'length_histogram': pd.DataFrame({'Awal': edges[:-1], 'Akhir': edges[1:], 'Ulasan': lengths}),
    }
