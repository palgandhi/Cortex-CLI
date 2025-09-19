# cortex/nlp/dynamic_parser.py
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import warnings

class DynamicNLPParser:
    def __init__(self):
        # Load a lightweight, pre-trained MiniLM model. This runs locally.
        warnings.filterwarnings("ignore")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            print("Please ensure you have internet access on first run.")
            print("You may also need to run: `pip install sentence-transformers`")
            self.model = None

        self.problem_types = {
            "regression": ["predict continuous values", "predict house prices", "predict salary", "predict income", "predict cost", "forecast sales"],
            "classification": ["predict a category", "classify text", "find spam", "categorize documents"],
            "reinforcement_learning": ["train an agent", "solve an environment", "get a reward"],
            "clustering": ["group data", "find clusters in data", "unsupervised learning", "segment customers"],
            "image_recognition": ["identify objects in images", "classify images", "detect faces", "recognize pictures"]
        }
        self.problem_type_vectors = self._create_embeddings(self.problem_types)

    def _create_embeddings(self, problem_dict):
        embeddings_dict = {}
        if self.model:
            for key, sentences in problem_dict.items():
                embeddings_dict[key] = self.model.encode(sentences)
        return embeddings_dict

    def parse_user_input(self, user_text):
        if not self.model:
            return None

        user_vector = self.model.encode([user_text])[0]
        best_match = {"problem_type": None, "score": 0}

        for problem_type, vectors in self.problem_type_vectors.items():
            for vector in vectors:
                similarity = 1 - cosine(user_vector, vector)
                if similarity > best_match["score"]:
                    best_match["score"] = similarity
                    best_match["problem_type"] = problem_type
        
        # Use a confidence threshold
        if best_match["score"] > 0.4:
            return best_match["problem_type"]
        
        return None