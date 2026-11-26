"""
src/feature_extraction.py
Pipeline Feature Extraction dan Dataset Splitting untuk Analisis Sentimen Ulasan GTA VI.
Sesuai panduan PRD & instruksi Phase 3:
1. Memuat dataset bersih data/processed/processed_reviews.csv
2. Menggunakan clean_text sebagai fitur teks, sentiment sebagai label target
3. Mengecualikan kolom non-fitur (score, review_id, platform, published_date)
4. Grouped stratified split ~80:20 berdasarkan clean_text, random_state=42
5. TF-IDF Vectorizer (max_features=20000, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)
   - Fitted HANYA pada X_train
   - Transformasi pada X_test tanpa refit
   - Tanpa stopword removal bahasa Inggris (multilingual)
6. Menyimpan artefak:
   - models/grouped_v1/tfidf_vectorizer.pkl
   - data/processed/grouped_v1/{X_train_tfidf,X_test_tfidf}.npz
   - data/processed/grouped_v1/{y_train,y_test}.csv
   - data/processed/grouped_v1/{split_membership.csv,manifest.json}
7. Validasi dan pemeriksaan integritas data
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from src.artifacts import environment, sha256

def run_feature_extraction(
    processed_path: str = "data/processed/processed_reviews.csv",
    models_dir: str = "models/grouped_v1",
    processed_dir: str = "data/processed/grouped_v1",
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 20000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True
):
    # Never overwrite existing baseline or an earlier versioned experiment.
    output_paths = [Path(models_dir) / 'tfidf_vectorizer.pkl'] + [
        Path(processed_dir) / name for name in
        ['X_train_tfidf.npz', 'X_test_tfidf.npz', 'y_train.csv', 'y_test.csv',
         'split_membership.csv', 'manifest.json']
    ]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Artifact sudah ada. Validasi artifact atau gunakan direktori versi baru.")
    if not 0 < test_size <= 0.5:
        raise ValueError("test_size harus dalam rentang (0, 0.5]")
    n_splits = round(1 / test_size)
    if not np.isclose(test_size, 1 / n_splits):
        raise ValueError("Grouped split memerlukan test_size=1/k untuk integer k >= 2")
    print("=" * 65)
    print("PHASE 3 — FEATURE EXTRACTION & DATASET SPLITTING PIPELINE")
    print("=" * 65)

    # 1. Dataset Preparation
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"File processed dataset tidak ditemukan: {processed_path}")

    print(f"\n[1] Membaca processed dataset dari: {processed_path}")
    df = pd.read_csv(processed_path)
    print(f"    Total data: {len(df):,} baris")
    print(f"    Kolom tersedia: {list(df.columns)}")

    # Verifikasi kolom dan non-empty values
    required = {'clean_text', 'sentiment', 'review_id'}
    if not required.issubset(df.columns):
        raise ValueError(f"Kolom wajib tidak lengkap: {required - set(df.columns)}")

    # Handle missing / null di clean_text atau sentiment jika ada
    null_texts = df['clean_text'].isnull().sum()
    null_labels = df['sentiment'].isnull().sum()
    print(f"    Null pada clean_text : {null_texts}")
    print(f"    Null pada sentiment  : {null_labels}")
    if null_texts > 0 or null_labels > 0:
        raise ValueError("Dataset mengandung null; perbaiki dataset secara eksplisit sebelum splitting")
    if df['review_id'].isna().any() or df['clean_text'].str.strip().eq('').any():
        raise ValueError("review_id/clean_text kosong tidak diperbolehkan")

    # Pastikan tipe string
    df['clean_text'] = df['clean_text'].astype(str)

    # Validasi kelas target yang diharapkan
    valid_classes = {'Negative', 'Neutral', 'Positive'}
    actual_classes = set(df['sentiment'].unique())
    print(f"    Kelas target terdeteksi: {actual_classes}")
    if actual_classes != valid_classes:
        raise ValueError(f"Kelas target tidak sesuai: {actual_classes}")

    # Fitur eksplisit & pemisahan target (EKSKLUSI score, review_id, platform, published_date)
    X = df['clean_text']
    y = df['sentiment']
    print("    Fitur input teks : 'clean_text'")
    print("    Target label     : 'sentiment'")
    print("    Fitur hanya teks; identitas kelompok dipakai untuk mencegah duplikasi lintas split")

    # 2. Grouped split; fold pertama dipilih sebelum evaluasi model.
    print(f"\n[2] Grouped stratified split (target test={test_size:.1%}, random_state={random_state})...")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # First fold is fixed in advance; never select a fold based on model results.
    # Conflicting labels stay unchanged, but identical text always stays together.
    train_indices, test_indices = next(splitter.split(X, y, groups=X))
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    if set(X_train) & set(X_test):
        raise ValueError("Teks identik ditemukan lintas train/test")

    n_train = len(X_train)
    n_test = len(X_test)
    n_total = n_train + n_test
    print(f"    Jumlah data Training : {n_train:>6,} ({n_train / n_total * 100:.1f}%)")
    print(f"    Jumlah data Testing  : {n_test:>6,} ({n_test / n_total * 100:.1f}%)")

    # Distribusi sentimen
    print("\n    Distribusi Sentimen pada Training Set:")
    train_dist = y_train.value_counts()
    train_pct = y_train.value_counts(normalize=True) * 100
    for label in ['Negative', 'Neutral', 'Positive']:
        print(f"      - {label:<8}: {train_dist[label]:>6,} ({train_pct[label]:5.2f}%)")

    print("\n    Distribusi Sentimen pada Testing Set:")
    test_dist = y_test.value_counts()
    test_pct = y_test.value_counts(normalize=True) * 100
    for label in ['Negative', 'Neutral', 'Positive']:
        print(f"      - {label:<8}: {test_dist[label]:>6,} ({test_pct[label]:5.2f}%)")

    # Verifikasi seluruh kelas ada di kedua split
    if set(y_train.unique()) != valid_classes or set(y_test.unique()) != valid_classes:
        raise ValueError("Ada kelas yang hilang di salah satu split")

    # 3. TF-IDF Feature Extraction
    print("\n[3] Konfigurasi TfidfVectorizer:")
    print(f"    - max_features : {max_features}")
    print(f"    - ngram_range  : {ngram_range}")
    print(f"    - min_df       : {min_df}")
    print(f"    - max_df       : {max_df}")
    print(f"    - sublinear_tf : {sublinear_tf}")
    print("    - stop_words   : None (Multilingual Dataset - Preserved)")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf
    )

    print("\n    Fitting TF-IDF Vectorizer HANYA pada X_train (mencegah data leakage)...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"    Dimensi X_train_tfidf: {X_train_tfidf.shape}")
    vocab_size = len(vectorizer.vocabulary_)
    print(f"    Jumlah kosakata (vocabulary) terbentuk: {vocab_size:,}")

    print("\n    Transformasi X_test menggunakan vectorizer yang telah di-fit...")
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"    Dimensi X_test_tfidf : {X_test_tfidf.shape}")

    # Validasi konsistensi dimensi
    assert X_train_tfidf.shape[1] == X_test_tfidf.shape[1], "Dimensi fitur train dan test tidak sama!"
    assert X_train_tfidf.shape[0] == len(y_train), "Jumlah baris X_train dan y_train tidak cocok!"
    assert X_test_tfidf.shape[0] == len(y_test), "Jumlah baris X_test dan y_test tidak cocok!"

    # 4. Save Artifacts
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    x_train_path = os.path.join(processed_dir, "X_train_tfidf.npz")
    x_test_path = os.path.join(processed_dir, "X_test_tfidf.npz")
    y_train_path = os.path.join(processed_dir, "y_train.csv")
    y_test_path = os.path.join(processed_dir, "y_test.csv")

    print("\n[4] Menyimpan Artefak:")
    # Vectorizer
    joblib.dump(vectorizer, vec_path)
    print(f"    [+] Saved: {vec_path} ({os.path.getsize(vec_path) / 1024 / 1024:.2f} MB)")

    # Sparse Matrices
    sparse.save_npz(x_train_path, X_train_tfidf)
    print(f"    [+] Saved: {x_train_path} ({os.path.getsize(x_train_path) / 1024 / 1024:.2f} MB)")
    sparse.save_npz(x_test_path, X_test_tfidf)
    print(f"    [+] Saved: {x_test_path} ({os.path.getsize(x_test_path) / 1024 / 1024:.2f} MB)")

    # Target Labels
    y_train.to_csv(y_train_path, index=False, header=True)
    print(f"    [+] Saved: {y_train_path} ({len(y_train):,} baris)")
    y_test.to_csv(y_test_path, index=False, header=True)
    print(f"    [+] Saved: {y_test_path} ({len(y_test):,} baris)")

    membership = pd.concat([
        pd.DataFrame({'row_index': indices, 'review_id': df.iloc[indices]['review_id'].to_numpy(),
                      'split': split, 'matrix_row': np.arange(len(indices))})
        for split, indices in [('train', train_indices), ('test', test_indices)]
    ], ignore_index=True)
    membership.to_csv(Path(processed_dir) / 'split_membership.csv', index=False)
    manifest = {
        'schema_version': 1, 'experiment': 'grouped_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_path': str(Path(processed_path).as_posix()),
        'dataset_sha256': sha256(processed_path), 'cleaning_version': 'legacy-v1',
        'split': {'method': 'StratifiedGroupKFold', 'group': 'clean_text', 'n_splits': n_splits,
                  'selected_fold': 0, 'shuffle': True, 'random_state': random_state,
                  'requested_test_size': test_size, 'actual_test_size': n_test / n_total},
        'tfidf': {'max_features': max_features, 'ngram_range': list(ngram_range), 'min_df': min_df,
                  'max_df': max_df, 'sublinear_tf': sublinear_tf, 'stop_words': None},
        'environment': environment(),
        'limitations': ['Provenance dan arti review_id belum terverifikasi independen.',
                        'Dataset processed legacy dipertahankan; cleaning multilingual memiliki keterbatasan.',
                        'Label adalah proksi rating; konflik label dipertahankan dalam satu kelompok.',
                        'Vektor kosong dipertahankan agar evaluasi mencakup seluruh split.'],
        'files': {str(path.as_posix()): sha256(path) for path in output_paths if path.name != 'manifest.json'},
        'source_files': {path: sha256(path) for path in
                         ['src/feature_extraction.py', 'src/preprocessing.py', 'src/artifacts.py']},
    }
    (Path(processed_dir) / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    # 5. Sampel Fitur N-gram Teratas
    feature_names = vectorizer.get_feature_names_out()
    print("\n[5] Contoh fitur unigram dan bigram teratas dalam vocabulary:")
    sample_features = feature_names[:15].tolist()
    print(f"    Contoh n-gram: {sample_features}")

    print("\n" + "=" * 65)
    print("PHASE 3 SELESAI DENGAN SUKSES!")
    print("=" * 65)

    return {
        'n_train': n_train,
        'n_test': n_test,
        'vocab_size': vocab_size,
        'X_train_shape': X_train_tfidf.shape,
        'X_test_shape': X_test_tfidf.shape,
        'train_dist': train_dist.to_dict(),
        'test_dist': test_dist.to_dict()
    }

if __name__ == "__main__":
    run_feature_extraction()
