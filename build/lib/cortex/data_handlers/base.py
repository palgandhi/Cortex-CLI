import os

class BaseDataHandler:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Abstract method to load the data. Must be implemented by subclasses."""
        raise NotImplementedError

    def detect_type(self):
        """Abstract method to detect the data type. Must be implemented by subclasses."""
        raise NotImplementedError