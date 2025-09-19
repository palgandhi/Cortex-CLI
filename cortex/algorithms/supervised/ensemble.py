from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from .classification import RandomForestClassifierModel
from .gbm import GradientBoostingClassifierModel
from ..base import BaseModel

class EnsembleClassifierModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ensemble Voting Classifier"

        # The base models are instantiated with default parameters
        # and trained within the VotingClassifier
        self.model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifierModel().model),
                ('gbm', GradientBoostingClassifierModel().model)
            ],
            voting='hard' # 'hard' voting uses predicted class labels
        )
        # No param_grid for this example
        self.param_grid = {}

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return {"Accuracy": accuracy}