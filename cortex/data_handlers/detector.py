# cortex/data_handlers/detector.py
# ... existing imports
from .tabular import TabularDataHandler
from .image import ImageDataHandler
from .text import TextDataHandler
from .environment import EnvironmentDataHandler

HANDLERS = [
    TabularDataHandler,
    ImageDataHandler,
    TextDataHandler,
    EnvironmentDataHandler,
]

def detect_dataset_type(file_path):
    """
    Detects the type of dataset and returns the appropriate handler.
    """
    for handler_class in HANDLERS:
        try:
            handler = handler_class(file_path)
            if handler.detect_type() is not None:
                return handler
        except Exception:
            # Silently ignore errors from handlers that fail to detect the type.
            # This allows the next handler in the list to be tried.
            continue

    return None