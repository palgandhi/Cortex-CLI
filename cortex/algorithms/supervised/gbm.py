from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from ..base import BaseModel

class GradientBoostingClassifierModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Gradient Boosting Classifier"
        self.model = GradientBoostingClassifier(**self.hyperparameters)
        self.param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return {"Accuracy": accuracy}