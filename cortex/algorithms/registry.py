from .supervised.regression import LinearRegressionModel
from .supervised.classification import RandomForestClassifierModel
from .supervised.text_models import TextClassifierModel
from .unsupervised.clustering import KMeansModel
from .reinforcement_learning.q_learning import QLearningAgent
from .supervised.xgboost import XGBoostClassifierModel
from .supervised.lightgbm import LightGBMClassifierModel
from .supervised.ensemble import EnsembleClassifierModel
from .supervised.xgboost_regressor import XGBoostRegressorModel # New import
from .supervised.lightgbm_regressor import LightGBMRegressorModel # New import

MODEL_REGISTRY = {
    "regression": [
        {"name": "XGBoost Regressor", "class": XGBoostRegressorModel, "description": "A powerful gradient boosting model for continuous values."},
        {"name": "LightGBM Regressor", "class": LightGBMRegressorModel, "description": "A fast and efficient gradient boosting model for continuous values."},
        {"name": "Linear Regression", "class": LinearRegressionModel, "description": "A simple, fast model for predicting continuous values."},
    ],
    "classification": [
        {"name": "XGBoost Classifier", "class": XGBoostClassifierModel, "description": "A powerful gradient boosting model, often a top choice in hackathons."},
        {"name": "LightGBM Classifier", "class": LightGBMClassifierModel, "description": "A fast and efficient gradient boosting model, ideal for large datasets."},
        {"name": "Random Forest Classifier", "class": RandomForestClassifierModel, "description": "An ensemble model that handles non-linear data well."},
        {"name": "Ensemble Voting Classifier", "class": EnsembleClassifierModel, "description": "Combines predictions from multiple models for improved accuracy."},
    ],
    "text_classification": [
        {"name": "Multinomial Naive Bayes", "class": TextClassifierModel, "description": "A probabilistic classifier suitable for text data."},
    ],
    "clustering": [
        {"name": "K-Means", "class": KMeansModel, "description": "A popular algorithm for finding groups in data."},
    ],
    "reinforcement_learning": [
        {"name": "Q-Learning Agent", "class": QLearningAgent, "description": "A classic algorithm for tabular environments."},
    ]
}

def get_suggested_models(problem_type):
    return MODEL_REGISTRY.get(problem_type, [])