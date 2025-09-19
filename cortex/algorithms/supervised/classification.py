from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ..base import BaseModel

class RandomForestClassifierModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Random Forest Classifier"
        self.model = RandomForestClassifier(**self.hyperparameters)
        self.param_grid = {
            'n_estimators': [50, 100, 200],  # Number of trees in the forest
            'max_depth': [None, 10, 20, 30]  # Maximum depth of the tree
        }

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        
        # Calculate multiple metrics for classification
        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, predictions, average='weighted', zero_division=0),
            "F1-Score": f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
        return metrics