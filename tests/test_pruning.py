from unittest import TestCase
import numpy as np
from sklearn.metrics import accuracy_score
from proactive_forest.estimator import ProactiveForestClassifier
from proactive_forest.pruning import AccuracyPruning
from sklearn.preprocessing import LabelEncoder


class AccuracyPruningTest(TestCase):
    def setUp(self):
        self.pruning = AccuracyPruning()
        self.X = np.array(['A', 'B', 'A', 'B', 'B', 'C',
                          'A', 'C', 'B']).reshape((3, 3))
        self.y = np.array([1, 1, 0])

        self.predictor = ProactiveForestClassifier()
        self.predictor._encoder = LabelEncoder()
        self.predictor._encoder.fit(self.y)
        self.predictor.fit(self.X, self.y)

    def tearDown(self):
        pass

    def test_pruning(self):
        initial_predictions = self.predictor.predict(self.X)
        initial_accuracy = accuracy_score(self.y, initial_predictions)

        initial_len, final_len = self.pruning.pruning(
            self.predictor, self.X, self.y)

        # Verificar que se hay menos árboles que los iniciales
        self.assertLess(final_len, initial_len)

        final_predictions = self.predictor.predict(self.X)
        final_accuracy = accuracy_score(self.y, final_predictions)

        # Verificar que la eficacia no disminuya
        self.assertGreaterEqual(final_accuracy, initial_accuracy)

    def test_pruning_exception(self):
        x = np.array(['A', 'B', 'A', 'C', 'A', 'A']).reshape((2, 3))
        y = [1, 0]

        self.decision_forest._encoder = mock.MagicMock(spec=LabelEncoder)
        self.decision_forest._encoder.transform.return_value = [1, 0]

        with self.assertRaises(ValueError):
            self.decision_forest.diversity_measure(x, y, diversity='kappa')
