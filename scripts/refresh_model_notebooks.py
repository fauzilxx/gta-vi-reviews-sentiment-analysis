"""Create Phase 4/5 notebooks from actual saved models and evaluation outputs.

Run from repository root after training/evaluation. No fitting is performed.
"""

import contextlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.refresh_notebooks import SETUP, code, markdown


def refresh():
    notebooks = {
        '04_model_training.ipynb': [
            markdown('# Phase 4 — Model training dan persistence\n'
                     'Model dilatih melalui `python -m src.train` menggunakan artifact grouped_v1. '
                     'Notebook menampilkan hasil training yang sudah tersimpan, tanpa training ulang. '
                     'PRD.md menjadi acuan; hyperparameter baseline dibekukan sebelum prediksi test.\n'),
            code(SETUP),
            code("plan = json.loads((ROOT / 'reports/grouped_v1/model_plan.json').read_text(encoding='utf-8'))\n"
                 "print(json.dumps(plan, indent=2))\n"),
            code("training = json.loads((ROOT / 'models/grouped_v1/training_manifest.json').read_text(encoding='utf-8'))\n"
                 "for name, info in training['models'].items():\n"
                 "    print(name, 'seconds=', round(info['training_seconds'], 3),\n"
                 "          'n_iter=', info['n_iter'], 'warnings=', info['warnings'],\n"
                 "          'reload_identical=', info['reload_training_predictions_identical'])\n"),
            markdown('## Kebijakan eksperimen\n'
                     'Ketiga model memakai matriks dan label training yang sama. '
                     'LR dan SVM memakai class_weight=balanced yang dihitung dari training. '
                     'Naive Bayes memakai prior empiris. Tidak ada tuning memakai test. '
                     'Model Joblib dilengkapi checksum, konfigurasi, dan versi environment.\n'),
        ],
        '05_model_evaluation.ipynb': [
            markdown('# Phase 5 — Evaluasi akhir dan perbandingan\n'
                     'Angka dibaca dari hasil prediksi model tersimpan pada seluruh test set grouped_v1. '
                     'Prioritas pemeringkatan adalah Macro F1. Precision/Recall tabel utama memakai macro average.\n'),
            code(SETUP),
            code("metrics = pd.read_csv(ROOT / 'reports/grouped_v1/metrics.csv')\n"
                 "print(metrics.to_string(index=False))\n"
                 "per_class = pd.read_csv(ROOT / 'reports/grouped_v1/per_class_metrics.csv')\n"
                 "print(per_class.to_string(index=False))\n"),
            code("result = json.loads((ROOT / 'reports/grouped_v1/evaluation.json').read_text(encoding='utf-8'))\n"
                 "for name, info in result['models'].items():\n"
                 "    print(name, 'confusion matrix (actual rows, predicted columns):', result['classes'])\n"
                 "    print(info['confusion_matrix'])\n"
                 "print('Majority baseline:', result['majority_baseline'])\n"
                 "print('Zero-vector test rows:', result['zero_vector_test_rows'])\n"),
            code("verification = json.loads((ROOT / 'reports/grouped_v1/evaluation_verification.json').read_text(encoding='utf-8'))\n"
                 "print(json.dumps({key: verification[key] for key in ['test_rows', 'models', 'feature_manifest_unchanged']}, indent=2))\n"),
            markdown('## Grafik dari prediksi aktual\n\n'
                     '![Perbandingan model](../reports/grouped_v1/figures/model_comparison.png)\n\n'
                     '![Confusion matrix](../reports/grouped_v1/figures/confusion_matrices.png)\n\n'
                     '![F1 per kelas](../reports/grouped_v1/figures/per_class_f1.png)\n'),
            markdown('## Keterbatasan\n'
                     'Evaluasi menggunakan satu held-out test. Label berasal dari rating, '
                     'dan subset deduplikasi serta cleaning lama tetap dipakai. '
                     'Vektor kosong tidak dihapus dari evaluasi. '
                     'Test yang telah dievaluasi tidak digunakan untuk tuning berikutnya. '
                     'Peringkat tidak menyatakan signifikansi statistik.\n'),
        ],
    }
    for filename, cells in notebooks.items():
        path = Path('notebooks') / filename
        if path.exists():
            raise FileExistsError(f'Notebook sudah ada: {path}')
        namespace, execution_count = {'__name__': '__notebook__'}, 0
        for cell in cells:
            if cell['cell_type'] == 'code':
                execution_count += 1
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(compile(''.join(cell['source']), filename, 'exec'), namespace)
                cell['execution_count'] = execution_count
                if output.getvalue():
                    cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout',
                                        'text': output.getvalue().splitlines(keepends=True)}]
        notebook = {'cells': cells, 'metadata': {
            'kernelspec': {'display_name': 'Python (.venv)', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
            'execution_note': 'Cells executed in order with Python exec; stdout captured by scripts/refresh_model_notebooks.py.'},
            'nbformat': 4, 'nbformat_minor': 4}
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        print('Notebook dengan output aktual:', path)


if __name__ == '__main__':
    refresh()
