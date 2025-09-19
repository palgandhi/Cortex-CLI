import xgboost as xgb
from sklearn.metrics import accuracy_score
from ..base import BaseModel

class XGBoostClassifierModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "XGBoost Classifier"
        self.model = xgb.XGBClassifier(**self.hyperparameters)
        self.param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01]
        }

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return {"Accuracy": accuracy}