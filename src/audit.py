"""Read existing data/artifacts and write reproducible EDA and diagnostic reports.

Run from the project root: python -m src.audit
No preprocessing, fitting, or artifact replacement is performed.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

from src.artifacts import environment, sha256, validate_artifacts

SOURCE_URL = "https://www.kaggle.com/datasets/ektarr/gta-vi-user-reviews/data"


def describe_dataset(df):
    result = {
        'rows': len(df), 'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(), 'missing': df.isna().sum().to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
        'date_range': [str(df.published_date.min()), str(df.published_date.max())],
    }
    for column in ['score', 'platform', 'sentiment']:
        if column in df:
            result[column + '_counts'] = df[column].value_counts().to_dict()
    for column in ['review_id', 'review_text', 'clean_text']:
        if column in df:
            result[column + '_duplicate_excess'] = int(df[column].duplicated().sum())
    return result


def build_reports():
    reports = Path('reports')
    reports.mkdir(exist_ok=True)
    raw = pd.read_csv('data/raw/gta_vi_reviews.csv')
    original = pd.read_csv('reviews.csv')
    processed_path = 'data/processed/processed_reviews.csv'
    df = pd.read_csv(processed_path)
    source_differences = {column: int(original[column].ne(raw[column]).sum()) for column in raw.columns}
    id_groups = raw.groupby('review_id').agg(
        rows=('review_id', 'size'), platforms=('platform', 'nunique'),
        distinct_texts=('review_text', 'nunique'), distinct_scores=('score', 'nunique'))
    id_groups[id_groups.rows > 1].to_csv(reports / 'duplicate_id_groups.csv')
    conflicting = df.groupby('clean_text').sentiment.nunique()
    df[df.clean_text.isin(conflicting[conflicting > 1].index)].to_csv(
        reports / 'normalized_label_conflicts.csv', index=False)
    expected_labels = np.select([df.score <= 4, df.score <= 7], ['Negative', 'Neutral'], default='Positive')
    baseline = json.loads((reports / 'baseline_artifacts.json').read_text(encoding='utf-8'))
    baseline_intact = all(sha256(path) == item['sha256'] for path, item in baseline['files'].items())
    if not baseline_intact:
        raise ValueError('Artifact baseline berubah')
    train, test = train_test_split(np.arange(len(df)), test_size=.2, stratify=df.sentiment, random_state=42)
    baseline_summary, zero_rows = {}, []
    configs = [('baseline', 'data/processed', 'models', {'train': train, 'test': test})]
    grouped_dir = Path('data/processed/grouped_v1')
    if (grouped_dir / 'manifest.json').exists():
        validation = validate_artifacts(processed_path, grouped_dir, 'models/grouped_v1')
        membership = pd.read_csv(grouped_dir / 'split_membership.csv')
        positions = {name: membership[membership.split == name].sort_values('matrix_row').row_index.to_numpy()
                     for name in ['train', 'test']}
        configs.append(('grouped_v1', str(grouped_dir), 'models/grouped_v1', positions))
    else:
        validation = None
    for experiment, folder, models, positions in configs:
        vectorizer = joblib.load(Path(models) / 'tfidf_vectorizer.pkl')
        for split, indices in positions.items():
            matrix = sparse.load_npz(Path(folder) / f'X_{split}_tfidf.npz')
            labels = pd.read_csv(Path(folder) / f'y_{split}.csv').sentiment
            if not np.array_equal(labels.to_numpy(), df.iloc[indices].sentiment.to_numpy()):
                raise ValueError('Label tidak cocok dengan split')
            empty = np.flatnonzero(matrix.getnnz(axis=1) == 0)
            for matrix_row in empty:
                row = df.iloc[indices[matrix_row]]
                tokens = vectorizer.build_analyzer()(row.clean_text)
                zero_rows.append({'experiment': experiment, 'split': split, 'matrix_row': int(matrix_row),
                                  'row_index': int(indices[matrix_row]), 'review_id': row.review_id,
                                  'sentiment': row.sentiment, 'analyzer_tokens': len(tokens),
                                  'reason': 'no_tokens' if not tokens else 'all_tokens_outside_vocabulary',
                                  'text_preview': row.clean_text[:180]})
            if experiment == 'baseline':
                baseline_summary[split] = {'shape': list(matrix.shape), 'zero_vectors': len(empty),
                                           'class_counts': labels.value_counts().to_dict()}
    # Prepare human review using training data only. Blank annotations are intentional.
    training = df.iloc[configs[-1][3]['train']]
    review_sample = pd.concat([
        group.sample(n=min(20, len(group)), random_state=42)
        for _, group in training.groupby('sentiment')
    ])[['review_id', 'review_text', 'score', 'sentiment']].copy()
    review_sample['human_sentiment'] = ''
    review_sample['review_notes'] = ''
    review_sample.to_csv(reports / 'label_review_sample.csv', index=False)
    pd.DataFrame(zero_rows).to_csv(reports / 'zero_vectors.csv', index=False)
    overlap = set(df.clean_text.iloc[train]) & set(df.clean_text.iloc[test])
    baseline_summary['overlap_clean_text'] = len(overlap)
    baseline_summary['test_rows_with_train_text'] = int(df.clean_text.iloc[test].isin(overlap).sum())
    summary = {
        'provenance': {'url': SOURCE_URL, 'provided_by': 'user', 'version': None, 'license': None,
                       'verification': 'URL supplied by user; Kaggle page had no readable documentation in browsing tool.',
                       'review_id_semantics': 'unverified; repeated IDs occur across platforms'},
        'raw': describe_dataset(raw), 'processed': describe_dataset(df),
        'source_column_difference_counts': source_differences,
        'duplicate_id_groups': {'total': int((id_groups.rows > 1).sum()),
                                'different_texts': int((id_groups.distinct_texts > 1).sum()),
                                'different_scores': int((id_groups.distinct_scores > 1).sum())},
        'normalized_label_conflict_groups': int((conflicting > 1).sum()),
        'rating_label_mismatches': int(np.count_nonzero(df.sentiment.to_numpy() != expected_labels)),
        'baseline_artifacts_unchanged': baseline_intact, 'baseline': baseline_summary,
        'grouped_v1': validation, 'environment': environment(),
    }
    (reports / 'dataset_audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    make_figures(raw, df, reports)
    return summary


def make_figures(raw, df, reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = reports / 'figures'
    folder.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    raw.score.value_counts().sort_index().plot.bar(ax=axes[0, 0], title='Raw rating distribution')
    df.sentiment.value_counts().reindex(['Negative', 'Neutral', 'Positive']).plot.bar(
        ax=axes[0, 1], title='Processed rating-derived sentiment')
    raw.platform.value_counts().plot.bar(ax=axes[1, 0], title='Raw platform distribution')
    df.clean_text.str.len().clip(upper=3000).plot.hist(
        ax=axes[1, 1], bins=50, title='Clean text length (characters; clipped at 3000)')
    axes[1, 1].set_xlabel('Characters')
    figure.tight_layout()
    figure.savefig(folder / 'dataset_overview.png', dpi=150)
    plt.close(figure)


if __name__ == '__main__':
    print(json.dumps(build_reports(), indent=2, ensure_ascii=False))
