import spacy
from fuzzywuzzy import fuzz

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    exit()

# Define keywords with synonyms for better matching
INTENT_KEYWORDS = {
    "predict": ["predict", "prediction", "forecast", "classify", "recognition"],
    "cluster": ["cluster", "group", "segment", "unsupervised"],
    "analyze": ["analyze", "explore", "describe"],
    "categorize": ["categorize", "identify", "determine", "type", "label"]
}

PROBLEM_TYPE_KEYWORDS = {
    "regression": ["prices", "value", "cost", "sales", "revenue", "amount"],
    "classification": ["category", "type", "class", "spam", "fraud", "scam"],
    "image_recognition": ["images", "photos", "pictures"],
    "text_classification": ["text", "document", "message", "email"],
    "clustering": ["group", "cluster", "unsupervised"] # New keywords for problem type
}

def parse_user_intent(text):
    """
    Parses natural language text to detect user intent and problem type using fuzzy matching.
    """
    doc = nlp(text.lower())
    
    intent = None
    problem_type = None

    # First, try to detect problem type
    for user_word in text.split():
        for problem_name, keywords in PROBLEM_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if fuzz.ratio(user_word, keyword) > 85:
                    problem_type = problem_name
                    break
            if problem_type:
                break
        if problem_type:
                break

    # Now, try to detect intent
    for user_word in text.split():
        for intent_name, keywords in INTENT_KEYWORDS.items():
            for keyword in keywords:
                if fuzz.ratio(user_word, keyword) > 85:
                    intent = intent_name
                    break
            if intent:
                break
        if intent:
            break
    
    # Simple rule for when intent is "cluster" but problem type is not set
    if intent == "cluster" and problem_type is None:
        problem_type = "clustering"

    return {"intent": intent, "problem_type": problem_type}