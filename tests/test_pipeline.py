"""Regression tests for the audit findings; no full-data refitting."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import validate_artifacts
from src.feature_extraction import run_feature_extraction
from src.preprocessing import assign_sentiment, clean_text, run_preprocessing


class PipelineTests(unittest.TestCase):
    def test_rating_boundaries_and_invalid_inputs(self):
        for score, expected in [(0, 'Negative'), (4, 'Negative'), (5, 'Neutral'), (7, 'Neutral'),
                                (8, 'Positive'), (10, 'Positive'), (np.int64(6), 'Neutral')]:
            self.assertEqual(assign_sentiment(score), expected)
        for value in [-1, 11, 4.9, np.nan, np.inf, None, True, 'invalid']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                assign_sentiment(value)

    def test_unicode_cleaning_is_versioned(self):
        self.assertEqual(clean_text("I don't like it"), 'i do not like it')
        self.assertEqual(clean_text('ไม่ดี', version='unicode-v2'), 'ไม่ดี')
        self.assertNotEqual(clean_text('ไม่ดี'), clean_text('ไม่ดี', version='unicode-v2'))
        self.assertEqual(clean_text('cafe\u0301', version='unicode-v2'), 'café')

    def test_preprocessing_refuses_existing_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'existing.csv'
            output.write_text('preserve', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                run_preprocessing(output_path=str(output))
            self.assertEqual(output.read_text(encoding='utf-8'), 'preserve')

    def test_grouped_artifacts_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for label, text in [('Negative', 'bad boring'), ('Neutral', 'average ordinary'), ('Positive', 'great fun')]:
                for index in range(15):
                    rows.append({'review_id': f'{label}-{index}', 'clean_text': f'{text} shared term{index}',
                                 'sentiment': label})
            # Identical text with a conflicting label must remain in one partition.
            rows.append({'review_id': 'conflict', 'clean_text': rows[0]['clean_text'], 'sentiment': 'Neutral'})
            source, data, models = root / 'data.csv', root / 'features', root / 'models'
            pd.DataFrame(rows).to_csv(source, index=False)
            arguments = dict(processed_path=str(source), processed_dir=str(data), models_dir=str(models),
                             min_df=1, max_df=1.0, max_features=100)
            run_feature_extraction(**arguments)
            result = validate_artifacts(source, data, models)
            self.assertEqual(result['overlap_clean_text'], 0)
            self.assertEqual(sum(result[s]['shape'][0] for s in ['train', 'test']), len(rows))
            with self.assertRaises(FileExistsError):
                run_feature_extraction(**arguments)
            labels = data / 'y_test.csv'
            labels.write_text('sentiment\nPositive\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_artifacts(source, data, models)


if __name__ == '__main__':
    unittest.main()
