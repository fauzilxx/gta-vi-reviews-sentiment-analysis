"""Refresh inspection notebooks with real captured outputs, never refit a pipeline.

Run from the repository root after python -m src.audit.
Original notebooks are preserved once under reports/baseline_notebooks/.
"""

import contextlib
import io
import json
import shutil
from pathlib import Path


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(keepends=True),
            'execution_count': None, 'outputs': []}


SETUP = """from pathlib import Path
import sys
import json
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / 'PRD.md').exists():
    ROOT = ROOT.parent
if not (ROOT / 'PRD.md').exists():
    raise RuntimeError('Buka notebook dari root proyek atau direktori notebooks')
sys.path.insert(0, str(ROOT))
from src.audit import describe_dataset
"""


def refresh():
    notebooks = {
        '01_eda.ipynb': [
            markdown('# Phase 1 — Inspeksi dataset dan EDA\n'
                     'Statistik dihitung dari CSV lokal. Asal yang diberikan pemilik proyek: '
                     'https://www.kaggle.com/datasets/ektarr/gta-vi-user-reviews/data. '
                     'Versi, lisensi, dan arti review_id belum terverifikasi independen.\n'),
            code(SETUP),
            code("raw = pd.read_csv(ROOT / 'data/raw/gta_vi_reviews.csv')\n"
                 "processed = pd.read_csv(ROOT / 'data/processed/processed_reviews.csv')\n"
                 "print(json.dumps({'raw': describe_dataset(raw), 'processed': describe_dataset(processed)}, indent=2))\n"),
            code("print('Panjang teks bersih (karakter):')\n"
                 "print(processed.clean_text.str.len().describe().to_string())\n"
                 "print('Distribusi label (%):')\n"
                 "print((processed.sentiment.value_counts(normalize=True) * 100).round(3).to_string())\n"),
            markdown('## Visualisasi\n'
                     'Grafik dibuat oleh `python -m src.audit` menggunakan file CSV aktual.\n\n'
                     '![Distribusi rating, label, platform, dan panjang teks](../reports/figures/dataset_overview.png)\n'),
            markdown('## Batasan interpretasi\n'
                     'Label merupakan proksi rating, bukan hasil anotasi manusia. '
                     'Porsi Neutral kecil. Dataset berisi teks multibahasa berdasarkan contoh yang diperiksa; '
                     'tidak ada kolom bahasa atau hasil deteksi bahasa yang terverifikasi. '
                     'ID berulang lintas platform memerlukan klarifikasi sebelum deduplikasi baru.\n'),
        ],
        '02_preprocessing.ipynb': [
            markdown('# Phase 2 — Verifikasi preprocessing yang sudah tersedia\n'
                     'Notebook membaca hasil lama, tanpa menjalankan ulang preprocessing. '
                     'Cleaning yang digunakan artifact saat ini adalah legacy-v1.\n'),
            code(SETUP),
            code("df = pd.read_csv(ROOT / 'data/processed/processed_reviews.csv')\n"
                 "print(json.dumps(describe_dataset(df), indent=2))\n"),
            code("from src.preprocessing import assign_sentiment, clean_text\n"
                 "mismatches = int(df.score.map(assign_sentiment).ne(df.sentiment).sum())\n"
                 "print('Label tidak sesuai rating:', mismatches)\n"
                 "assert mismatches == 0\n"
                 "print('Jumlah kelompok teks bersih dengan label bertentangan:',\n"
                 "      int(df.groupby('clean_text').sentiment.nunique().gt(1).sum()))\n"
                 "sample = df.sample(n=min(200, len(df)), random_state=42)\n"
                 "assert sample.review_text.map(clean_text).eq(sample.clean_text).all()\n"
                 "print('Cleaning legacy-v1 cocok untuk', len(sample), 'sampel deterministik')\n"),
            markdown('## Keputusan\n'
                     'Dataset lama dipertahankan. Konflik label tidak diubah secara otomatis. '
                     'unicode-v2 tersedia untuk eksperimen baru, tetapi belum digunakan pada dataset ini. '
                     'Jangan mencampurkan unicode-v2 dengan vectorizer legacy-v1. '
                     'Kebijakan penghapusan review_id lama masih memerlukan klarifikasi sumber.\n'),
        ],
        '03_feature_extraction.ipynb': [
            markdown('# Phase 3 — Verifikasi artifact TF-IDF berkelompok\n'
                     'Notebook menggunakan artifact tersimpan, tanpa fitting. '
                     'Fold pertama StratifiedGroupKFold (5 fold, shuffle, seed 42) '
                     'menempatkan clean_text identik di satu split.\n'),
            code(SETUP),
            code("manifest = json.loads((ROOT / 'data/processed/grouped_v1/manifest.json').read_text(encoding='utf-8'))\n"
                 "print(json.dumps({key: manifest[key] for key in ['experiment', 'cleaning_version', 'split', 'tfidf', 'environment']}, indent=2))\n"),
            code("import os\n"
                 "from src.artifacts import validate_artifacts\n"
                 "previous_cwd = Path.cwd()\n"
                 "try:\n"
                 "    os.chdir(ROOT)  # manifest menyimpan path relatif terhadap root proyek\n"
                 "    validation = validate_artifacts('data/processed/processed_reviews.csv',\n"
                 "                                    'data/processed/grouped_v1', 'models/grouped_v1')\n"
                 "finally:\n"
                 "    os.chdir(previous_cwd)\n"
                 "print(json.dumps(validation, indent=2))\n"),
            markdown('## Validasi dan keterbatasan\n'
                     'Validasi memeriksa checksum, cakupan baris, urutan label, hasil transformasi, '
                     'IDF training, dan overlap teks. Transform hanya untuk verifikasi di memori. '
                     'Vektor kosong dipertahankan; diagnosis ada di reports/zero_vectors.csv. '
                     'Vectorizer lama dan seluruh artifact baseline tetap tersedia.\n'),
        ],
    }
    for filename, cells in notebooks.items():
        path = Path('notebooks') / filename
        backup = Path('reports/baseline_notebooks') / filename
        if filename != '01_eda.ipynb' and path.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        namespace, execution_count = {'__name__': '__notebook__'}, 0
        for cell in cells:
            if cell['cell_type'] != 'code':
                continue
            execution_count += 1
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(compile(''.join(cell['source']), filename, 'exec'), namespace)
            cell['execution_count'] = execution_count
            if output.getvalue():
                cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': output.getvalue().splitlines(keepends=True)}]
        notebook = {'cells': cells, 'metadata': {
            'kernelspec': {'display_name': 'Python (.venv)', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
            'execution_note': 'Code cells executed in order with Python exec; stdout captured by scripts/refresh_notebooks.py.'},
            'nbformat': 4, 'nbformat_minor': 4}
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        print('Notebook tersimpan dengan output aktual:', path)


if __name__ == '__main__':
    refresh()
