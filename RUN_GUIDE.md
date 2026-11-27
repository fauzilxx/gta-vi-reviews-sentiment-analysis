# How to Run This Project (From Zero to Working Dashboard)

This guide walks through setting up the exact environment required to verify the experiments and run the Streamlit dashboard locally on Windows (PowerShell).

> **Important:** The saved models and manifests in `grouped_v1` strictly check the Python version (**3.13.0**) and exact package versions in `requirements.txt`, as well as SHA-256 checksums of the dataset and matrices. Using a different Python version (e.g., 3.14) will cause the integrity check to fail.

---

## 1. Install Python 3.13.0

Choose **one** of the two options below:

### Option A — Using `uv` (Recommended, fastest)
If you don't have `uv`, install it in PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
`uv` will automatically download **Python 3.13.0** when creating the virtual environment in Step 3.

### Option B — Official Python 3.13.0 Installer
1. Download **Python 3.13.0 (64-bit)** from [python.org/downloads/release/python-3130](https://www.python.org/downloads/release/python-3130/).
2. Run the installer and check **"Add python.exe to PATH"**.
3. Verify in PowerShell:
   ```powershell
   py -3.13 --version
   # Expected: Python 3.13.0
   ```

---

## 2. Clone and Enter the Project Folder

```powershell
git clone https://github.com/fauzilxx/gta-vi-reviews-sentiment-analysis.git
cd gta-vi-reviews-sentiment-analysis
```

---

## 3. Create `.venv` (Python 3.13.0) and Install Dependencies

### If using `uv`:
```powershell
uv venv --python 3.13.0 .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

### If using standard Python 3.13.0:
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 4. Rebuild Git-Ignored `.npz` Matrices & Fix Windows Line Endings (After Fresh Clone)

Because `.gitignore` excludes `data/processed/**/*.npz` (large sparse matrices) and Git on Windows may convert `LF` to `CRLF` on checkout, run this one-time setup script in PowerShell from the project root so every SHA-256 checksum matches `manifest.json` and `baseline_artifacts.json`:

```powershell
.\.venv\Scripts\python.exe -B -c '
import hashlib, io, json, subprocess
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Restore exact line endings for checksum-checked files
pp = Path("src/preprocessing.py")
pp.write_bytes(pp.read_bytes().replace(b"\r\n", b"\n"))

for csv_path, expected_sha in [
    ("data/processed/processed_reviews.csv", json.loads(Path("data/processed/grouped_v1/manifest.json").read_text(encoding="utf-8"))["dataset_sha256"]),
    ("data/raw/gta_vi_reviews.csv", json.loads(Path("reports/baseline_artifacts.json").read_text(encoding="utf-8"))["files"]["data/raw/gta_vi_reviews.csv"]["sha256"]),
    ("reviews.csv", json.loads(Path("reports/baseline_artifacts.json").read_text(encoding="utf-8"))["files"]["reviews.csv"]["sha256"]),
]:
    blob = subprocess.check_output(["git", "show", f"HEAD:{csv_path}"])
    if hashlib.sha256(blob).hexdigest() == expected_sha:
        Path(csv_path).write_bytes(blob)
    else:
        df_tmp = pd.read_csv(io.BytesIO(blob))
        Path(csv_path).write_bytes(df_tmp.to_csv(index=False, lineterminator="\r\n").encode("utf-8"))

# 2. Rebuild grouped_v1 TF-IDF .npz matrices
df = pd.read_csv("data/processed/processed_reviews.csv")
df["clean_text"] = df["clean_text"].astype(str)
membership = pd.read_csv("data/processed/grouped_v1/split_membership.csv")
train_texts = df.iloc[membership[membership["split"] == "train"].sort_values("matrix_row")["row_index"].to_numpy()]["clean_text"]
test_texts = df.iloc[membership[membership["split"] == "test"].sort_values("matrix_row")["row_index"].to_numpy()]["clean_text"]

vec_fit = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
sparse.save_npz("data/processed/grouped_v1/X_train_tfidf.npz", vec_fit.fit_transform(train_texts))
vec_grouped = joblib.load("models/grouped_v1/tfidf_vectorizer.pkl")
sparse.save_npz("data/processed/grouped_v1/X_test_tfidf.npz", vec_grouped.transform(test_texts))

# 3. Rebuild baseline TF-IDF .npz matrices (needed for src.audit)
tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df["sentiment"], random_state=42)
vec_base = joblib.load("models/tfidf_vectorizer.pkl")
sparse.save_npz("data/processed/X_train_tfidf.npz", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True).fit_transform(df["clean_text"].iloc[tr_idx]))
sparse.save_npz("data/processed/X_test_tfidf.npz", vec_base.transform(df["clean_text"].iloc[te_idx]))

print("Setup complete.")
'
```

---

## 5. Verify Everything Works

Run the unit/dashboard test suite and the read-only evaluation verifier:

```powershell
.\.venv\Scripts\python.exe -B -m src.evaluate --verify-only
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Expected output:
- `src.evaluate --verify-only` prints JSON with all checks `true` for `9,996` test rows.
- `unittest` finishes with `Ran 15 tests ... OK`.

---

## 6. Run the Streamlit Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Open **http://localhost:8501** in your browser. Press **Ctrl+C** in the terminal to stop the server.

### Dashboard Pages
1. **Overview** — project summary, dataset size, sentiment distribution, and label caveats.
2. **Exploratory data analysis** — rating, platform, sentiment, text length, and frequent terms.
3. **Model comparison** — Accuracy, Macro Precision, Macro Recall, Macro F1, and Weighted F1.
4. **Model evaluation** — interactive confusion matrices (counts & row-normalized) and classification reports.
5. **Sentiment prediction** — test live predictions across Multinomial Naive Bayes, Logistic Regression, and LinearSVC.

---

## 7. Optional Audit & Alignment Commands

- **Refresh dataset audit & EDA figure** (read-only on models/splits):
  ```powershell
  .\.venv\Scripts\python.exe -B -m src.audit
  ```
- **Recompute human-annotation vs. rating-proxy alignment:**
  ```powershell
  .\.venv\Scripts\python.exe -B -m src.label_alignment
  ```
