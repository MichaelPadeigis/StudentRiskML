import unittest
import pandas as pd
from train import build_pipeline, read_dataset, risk_level, select_data


class TestPipeline(unittest.TestCase):
    def test_risk_levels(self):
        self.assertEqual(risk_level(0.10), 'Low')
        self.assertEqual(risk_level(0.30), 'Moderate')
        self.assertEqual(risk_level(0.80), 'High')
        with self.assertRaises(ValueError):
            risk_level(0.5, 0.9, 0.4)

    def test_timing(self):
        df = pd.DataFrame({
            'Target': ['dropout', 'graduate', 'enrolled'],
            'Age at enrollment': [19, 20, 21],
            'Curricular units 1st sem (grade)': [10, 11, 12],
            'Curricular units 2nd sem (grade)': [9, 10, 11]
        })
        X, y = select_data(df)
        self.assertEqual(len(X), 2)
        self.assertEqual(X.columns.tolist(), ['Age at enrollment'])
        self.assertEqual(y.tolist(), [1, 0])
        X, _ = select_data(df, 'after_first_semester')
        self.assertEqual(len(X.columns), 2)
        self.assertNotIn('Curricular units 2nd sem (grade)', X.columns)

    def test_model(self):
        X = pd.DataFrame({
            'Age at enrollment': [18, 19, 20, 21, 22, 23, 24, 25],
            'Course': [171, 171, 171, 171, 9003, 9003, 9003, 9003]
        })
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        pipeline = build_pipeline('logistic_regression', X).fit(X, y)
        self.assertEqual(pipeline.predict_proba(X).shape, (8, 2))
        self.assertGreater(pipeline.predict_proba(X)[-1, 1], 0.50)

    def test_real_file(self):
        df = read_dataset('data/data.csv')
        self.assertEqual(len(df), 4424)
        self.assertEqual(df['Target'].value_counts()['dropout'], 1421)


if __name__ == '__main__':
    unittest.main()
