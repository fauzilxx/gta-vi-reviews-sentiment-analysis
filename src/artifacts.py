"""Checksums, metadata, and integrity checks for versioned experiments."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
    }


def validate_artifacts(processed_path, processed_dir, models_dir):
    """Verify saved row identity, labels, features, and training-only IDF; never fit."""
    processed_dir, models_dir = Path(processed_dir), Path(models_dir)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256(processed_path) != manifest["dataset_sha256"]:
        raise ValueError("Dataset berubah sejak artifact dibuat")
    for path, expected in manifest["files"].items():
        if sha256(path) != expected:
            raise ValueError(f"Checksum artifact tidak sesuai: {path}")
    df = pd.read_csv(processed_path)
    membership = pd.read_csv(processed_dir / "split_membership.csv")
    if len(membership) != len(df) or set(membership["row_index"]) != set(range(len(df))):
        raise ValueError("Split tidak mencakup setiap baris dataset tepat sekali")
    if membership["row_index"].duplicated().any() or set(membership["split"]) != {"train", "test"}:
        raise ValueError("Identitas baris/split tidak valid")
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
    result, texts = {}, {}
    for split in ["train", "test"]:
        members = membership[membership["split"] == split].sort_values("matrix_row")
        if not np.array_equal(members["matrix_row"], np.arange(len(members))):
            raise ValueError("Urutan baris matriks tidak valid")
        rows = df.iloc[members["row_index"].to_numpy()]
        if not np.array_equal(rows["review_id"].to_numpy(), members["review_id"].to_numpy()):
            raise ValueError("review_id tidak sesuai dengan dataset")
        labels = pd.read_csv(processed_dir / f"y_{split}.csv")["sentiment"]
        matrix = sparse.load_npz(processed_dir / f"X_{split}_tfidf.npz")
        if matrix.shape != (len(rows), len(vectorizer.vocabulary_)):
            raise ValueError("Dimensi matriks tidak sesuai")
        if not np.array_equal(labels.to_numpy(), rows["sentiment"].to_numpy()):
            raise ValueError("Label tidak sesuai urutan matriks")
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError("Nilai fitur tidak valid")
        delta = matrix - vectorizer.transform(rows["clean_text"])
        if delta.nnz and np.max(np.abs(delta.data)) > 1e-12:
            raise ValueError("Matriks berbeda dari transformasi vectorizer tersimpan")
        if split == "train":
            frequency = np.bincount(matrix.indices, minlength=matrix.shape[1])
            expected_idf = np.log((1 + matrix.shape[0]) / (1 + frequency)) + 1
            if not np.allclose(expected_idf, vectorizer.idf_, rtol=0, atol=1e-12):
                raise ValueError("IDF tidak sesuai training set")
        texts[split] = set(rows["clean_text"])
        result[split] = {"shape": list(matrix.shape), "zero_vectors": int((matrix.getnnz(axis=1) == 0).sum()),
                         "class_counts": labels.value_counts().to_dict()}
    overlap = len(texts["train"] & texts["test"])
    if overlap:
        raise ValueError(f"Terdapat {overlap} teks identik lintas split")
    result["overlap_clean_text"] = overlap
    return result
