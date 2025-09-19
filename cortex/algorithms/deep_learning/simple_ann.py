import torch
import torch.nn as nn
from cortex.algorithms.base import BaseModel # Inherit for core CLI compatibility

class BaseDeepLearningModel(BaseModel):
    """
    Abstract base class for all deep learning models in Cortex.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, dataloader):
        """Trains the model on the provided DataLoader."""
        raise NotImplementedError

    def evaluate(self, dataloader):
        """Evaluates the model on the provided DataLoader."""
        raise NotImplementedError

    def save(self, file_path):
        """Saves the model's state dictionary to a file."""
        if self.model:
            torch.save(self.model.state_dict(), file_path)
            print(f"Deep learning model saved to {file_path}")
        else:
            print("Model not trained yet. Cannot save.")