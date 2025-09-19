import os
import imghdr  # A standard library to determine the type of an image

from .base import BaseDataHandler

class ImageDataHandler(BaseDataHandler):
    def load_data(self):
        # For image datasets, we don't load all images into memory.
        # We just verify the path and file count.
        if os.path.isdir(self.file_path):
            files = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path)]
            self.data = [f for f in files if imghdr.what(f) is not None]
        else:
            # Handle a single image file if needed
            self.data = [self.file_path] if imghdr.what(self.file_path) is not None else []

    def detect_type(self):
        # Check if the path is a directory and contains image files.
        if os.path.isdir(self.file_path):
            files = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path)]
            image_files = [f for f in files if imghdr.what(f) is not None]
            if len(image_files) > 0:
                return "image"
        # Also check for a single image file
        elif imghdr.what(self.file_path) is not None:
            return "image"
        return None