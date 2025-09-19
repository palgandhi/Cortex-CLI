from .tabular import TabularDataHandler
from .image import ImageDataHandler
from .text import TextDataHandler
# We will add other handlers here later (e.g. TimeSeriesDataHandler)

# List all available handlers
HANDLERS = [
    TabularDataHandler,
    ImageDataHandler,
    TextDataHandler
]

def detect_dataset_type(file_path):
    """
    Detects the type of dataset and returns the appropriate handler.
    
    Args:
        file_path (str): The path to the dataset.
    
    Returns:
        BaseDataHandler: An instantiated data handler class, or None if detection fails.
    """
    for handler_class in HANDLERS:
        handler = handler_class(file_path)
        if handler.detect_type() is not None:
            return handler
    
    return None