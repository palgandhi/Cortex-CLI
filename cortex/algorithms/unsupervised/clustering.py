# cortex/algorithms/clustering.py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ..base import BaseModel

class KMeansModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "K-Means Clustering"
        # We define a simple param_grid for hyperparameter tuning.
        self.param_grid = {'n_clusters': [2, 3, 4]}
        self.model = KMeans(**self.hyperparameters)

    def train(self, X_train):
        """Trains the model without a target variable."""
        self.model.fit(X_train)

    def evaluate(self, X_test):
        """Evaluates the trained model using the silhouette score."""
        # Note: silhouette_score requires a pre-trained model to make predictions.
        labels = self.model.predict(X_test)
        score = silhouette_score(X_test, labels)
        return {"Silhouette Score": score}