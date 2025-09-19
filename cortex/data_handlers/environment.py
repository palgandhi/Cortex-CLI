import gymnasium as gym
from .base import BaseDataHandler

class EnvironmentDataHandler(BaseDataHandler):
    def __init__(self, env_id):
        # In this handler, the file_path is the environment ID
        super().__init__(env_id)
        self.env_id = env_id
        self.data = None  # The environment object

    def load_data(self):
        """Instantiates the specified gymnasium environment."""
        try:
            self.data = gym.make(self.env_id)
        except gym.error.UnregisteredEnvError:
            raise ValueError(f"Environment '{self.env_id}' not found.")
        except Exception as e:
            raise ValueError(f"Could not load environment '{self.env_id}': {e}")
        print(f"Successfully loaded environment: {self.env_id}")

    def detect_type(self, is_valid_id=False):
        """
        Detects if the input path is a valid gymnasium environment ID.
        A simple check is enough for now.
        """
        if is_valid_id:
            return "environment"

        # Simple heuristic for a valid env ID string
        if isinstance(self.env_id, str) and len(self.env_id) > 2 and '-' in self.env_id and not '.' in self.env_id:
            try:
                # We can't use gym.make() directly as it has side effects, so we use a different check.
                # This is just a heuristic. A real-world app would need a more robust check.
                return "environment"
            except Exception:
                pass
        return None