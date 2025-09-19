# cortex/data_handlers/text.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from .base import BaseDataHandler

class TextDataHandler(BaseDataHandler):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.vectorizer = CountVectorizer()
        self.X_transformed = None
        self.target_column = None

    def load_data(self):
        # We assume the data is a CSV with two columns: text and a label.
        self.data = pd.read_csv(self.file_path)
        self.data.columns = [col.lower() for col in self.data.columns] # Standardize column names

    def get_features_and_target(self, target_column):
        """
        Processes text data and returns features (X) and target (y).
        """
        if not self.data or target_column not in self.data.columns:
            raise ValueError("Data not loaded or target column not found.")
            
        text_column = [col for col in self.data.columns if col != target_column][0]
        
        X = self.data[text_column]
        y = self.data[target_column]
        
        self.X_transformed = self.vectorizer.fit_transform(X)
        self.target_column = target_column
        
        return self.X_transformed, y

    def detect_type(self):
        if self.file_path.endswith('.csv'):
            try:
                # Peek at the first few rows to check for string-heavy data
                df = pd.read_csv(self.file_path, nrows=5)
                # Check if at least one column is an object (string) type and has high cardinality
                is_text_data = any(df[col].dtype == 'object' and len(df[col].unique()) > 2 for col in df.columns)
                if is_text_data:
                    return "text"
            except Exception:
                return None
        return None