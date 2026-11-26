"""AppTest and inference regressions against the persisted experiment; no fitting."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.modeling import CLASSES, MODEL_NAMES
from src.predict import PROJECT_ROOT, load_model_bundle, predict_sentiment


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle()

    def test_prediction_matches_saved_test_row_for_every_model(self):
        membership = pd.read_csv(PROJECT_ROOT / 'data/processed/grouped_v1/split_membership.csv')
        first = membership[membership.split == 'test'].sort_values('matrix_row').iloc[0]
        dataset = pd.read_csv(PROJECT_ROOT / 'data/processed/processed_reviews.csv', usecols=['review_text', 'clean_text'])
        source = dataset.iloc[int(first.row_index)]
        saved = pd.read_csv(PROJECT_ROOT / 'reports/grouped_v1/predictions.csv').iloc[0]
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                result = predict_sentiment(source.review_text, name, self.bundle)
                self.assertEqual(result['clean_text'], source.clean_text)
                self.assertEqual(result['label'], saved[name])

    def test_invalid_and_unknown_vocabulary_input(self):
        for text in ['', '   ', '12345 !!! https://example.com', 'qzxvnonexistenttokenxx qzxvnonexistenttokenyy']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                predict_sentiment(text, 'linear_svm', self.bundle)
        with self.assertRaises(ValueError):
            predict_sentiment('great game', 'unknown', self.bundle)

    def test_missing_artifact_and_corrupted_checksum_fail(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(FileNotFoundError):
            load_model_bundle(folder)
        with patch('src.predict.sha256', return_value='incorrect'), self.assertRaises(ValueError):
            load_model_bundle()


class DashboardTests(unittest.TestCase):
    def new_app(self):
        return AppTest.from_file(str(PROJECT_ROOT / 'dashboard/app.py'), default_timeout=60).run()

    def assert_healthy(self, app):
        self.assertFalse(list(app.exception), [item.value for item in app.exception])
        self.assertFalse(list(app.error), [item.value for item in app.error])

    def test_every_page_runs_without_any_fit(self):
        with patch('sklearn.naive_bayes.MultinomialNB.fit', side_effect=AssertionError('No fit allowed')), \
             patch('sklearn.linear_model.LogisticRegression.fit', side_effect=AssertionError('No fit allowed')), \
             patch('sklearn.svm.LinearSVC.fit', side_effect=AssertionError('No fit allowed')), \
             patch('sklearn.feature_extraction.text.TfidfVectorizer.fit', side_effect=AssertionError('No fit allowed')), \
             patch('sklearn.feature_extraction.text.TfidfVectorizer.fit_transform', side_effect=AssertionError('No fit allowed')):
            app = self.new_app()
            self.assert_healthy(app)
            self.assertEqual(app.metric[0].value, '49,981')
            for page in ['eda', 'comparison', 'evaluation', 'prediction']:
                with self.subTest(page=page):
                    app.switch_page(f'app_pages/{page}.py').run()
                    self.assert_healthy(app)
            app.segmented_control(key='prediction_model').set_value('linear_svm')
            app.text_area(key='review_text').set_value('The story is engaging and the world is beautiful.')
            app.button(key='predict_submit').click().run()
            self.assert_healthy(app)
            self.assertTrue(any(item.value in CLASSES for item in app.subheader))

    def test_comparison_table_matches_saved_metrics(self):
        app = self.new_app().switch_page('app_pages/comparison.py').run()
        self.assert_healthy(app)
        table = app.dataframe[0].value.set_index('Model')
        expected = json.loads((PROJECT_ROOT / 'reports/grouped_v1/evaluation.json').read_text(encoding='utf-8'))
        for name, info in expected['models'].items():
            for column, metric in [('Accuracy', 'accuracy'), ('Precision macro', 'precision_macro'),
                                   ('Recall macro', 'recall_macro'), ('Macro F1', 'f1_macro'), ('Weighted F1', 'f1_weighted')]:
                self.assertAlmostEqual(table.loc[MODEL_NAMES[name], column], info['metrics'][metric] * 100)
        self.assertTrue(table.loc['Logistic Regression', 'Pilihan Macro F1'])

    def test_evaluation_switches_models_and_normalization(self):
        app = self.new_app().switch_page('app_pages/evaluation.py').run()
        expected = json.loads((PROJECT_ROOT / 'reports/grouped_v1/evaluation.json').read_text(encoding='utf-8'))
        for name in MODEL_NAMES:
            app.segmented_control(key='evaluation_model').set_value(name).run()
            self.assert_healthy(app)
            neutral = app.dataframe[0].value.set_index('Kelas / rata-rata').loc['Neutral']
            self.assertAlmostEqual(neutral['Recall'], expected['models'][name]['classification_report']['Neutral']['recall'] * 100)
        app.toggle(key='normalized_matrix').set_value(True).run()
        self.assert_healthy(app)
        spec = json.loads(app.get('vega_lite_chart')[0].proto.spec)
        self.assertEqual(spec['layer'][0]['encoding']['color']['scale']['domain'], [0, 1])

    def test_eda_word_count_control(self):
        app = self.new_app().switch_page('app_pages/eda.py').run()
        self.assert_healthy(app)
        app.segmented_control(key='word_limit').set_value(30).run()
        self.assert_healthy(app)
        self.assertEqual(len(app.get('vega_lite_chart')), 5)

    def test_prediction_form_success_empty_and_unknown_words(self):
        app = self.new_app().switch_page('app_pages/prediction.py').run()
        bundle = load_model_bundle()
        for name in MODEL_NAMES:
            text = 'I do not like this boring game and its terrible performance.'
            app.segmented_control(key='prediction_model').set_value(name)
            app.text_area(key='review_text').set_value(text)
            app.button(key='predict_submit').click().run()
            self.assert_healthy(app)
            expected = predict_sentiment(text, name, bundle)['label']
            self.assertIn(expected, [item.value for item in app.subheader])
        for text in ['', '12345 !!!', 'qzxvnonexistenttokenxx qzxvnonexistenttokenyy']:
            app.text_area(key='review_text').set_value(text)
            app.button(key='predict_submit').click().run()
            self.assert_healthy(app)
            self.assertTrue(list(app.warning))
            self.assertNotIn('HASIL PREDIKSI', [item.value for item in app.caption])

    def test_missing_artifact_shows_friendly_error(self):
        with patch('dashboard.data.get_bundle', side_effect=FileNotFoundError('missing')), \
             patch('logging.Logger.exception'):
            app = self.new_app()
        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.error), 1)
        self.assertIn('File eksperimen tidak tersedia', app.error[0].value)


if __name__ == '__main__':
    unittest.main()
