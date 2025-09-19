import joblib

class BaseModel:
    """
    Abstract base class for all machine learning models in Cortex.
    """
    def __init__(self, **kwargs):
        self.model = None
        self.name = "Unknown Model"
        self.hyperparameters = kwargs

    def train(self, X_train, y_train):
        """Trains the model on the provided data."""
        raise NotImplementedError

    def evaluate(self, X_test, y_test):
        """Evaluates the trained model and returns metrics."""
        raise NotImplementedError

    def save(self, file_path):
        """Saves the trained model to a file."""
        if self.model:
            joblib.dump(self.model, file_path)
            print(f"Model saved to {file_path}")
        else:
            print("Model not trained yet. Cannot save.")