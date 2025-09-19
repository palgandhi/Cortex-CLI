# tests/test_algorithms.py
import pytest
from cortex.algorithms.supervised.gbm import GradientBoostingClassifierModel
from sklearn.datasets import make_classification

def test_gbm_model_training():
    # Create a simple dummy dataset
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Instantiate and train the model
    model = GradientBoostingClassifierModel()
    model.train(X, y)

    # Check if the model has been trained successfully
    assert model.model is not None
    assert model.model.classes_ is not None

def test_gbm_model_evaluation():
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    model = GradientBoostingClassifierModel()
    model.train(X, y)

    # Evaluate the model
    metrics = model.evaluate(X, y)

    # Check that the evaluation metrics are returned and have the correct format
    assert "Accuracy" in metrics
    assert isinstance(metrics["Accuracy"], float)
    assert 0.0 <= metrics["Accuracy"] <= 1.0