import pandas as pd
from .base import BaseDataHandler

class TabularDataHandler(BaseDataHandler):
    def load_data(self):
        # We'll use pandas to load the data, as it's the standard for tabular data.
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith(('.xls', '.xlsx')):
            self.data = pd.read_excel(self.file_path)
        # Add more formats like Parquet, JSON, etc., later.

    def detect_type(self):
        # Basic check based on file extension. We can add more sophisticated checks later.
        if self.file_path.endswith(('.csv', '.xls', '.xlsx', '.parquet')):
            return "tabular"
        return None