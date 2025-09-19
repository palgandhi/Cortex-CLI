from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ..base import BaseModel

class TextClassifierModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Multinomial Naive Bayes (Text)"
        # Use a pipeline to combine vectorization and classification
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])
        # Define the parameter grid for tuning the pipeline
        self.param_grid = {
            'vectorizer__ngram_range': [(1, 1), (1, 2)],
            'classifier__alpha': [0.1, 1.0, 10.0]
        }

    def train(self, X_train, y_train):
        # The pipeline handles both vectorization and training
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        
        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, predictions, average='weighted', zero_division=0),
            "F1-Score": f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
        return metrics