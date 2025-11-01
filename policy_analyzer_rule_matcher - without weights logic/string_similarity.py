import re
import jellyfish
import string  # Import the string module


def normalize(s: str) -> str:
    """
    Cleans and standardizes a string value before comparison.
    - Converts to lowercase
    - Trims whitespace
    - Removes ALL punctuation
    - Removes common surrounding characters like quotes or curly braces
    """
    if s is None:
        return ""
    s = str(s).lower().strip()

    # This regex removes quotes or {} or "" from the very beginning or end
    s = re.sub(r'^[{"\']+|[}"\']+$', '', s)
    s = re.sub(r'^{""|""}$', '', s)

    # --- THIS IS THE FIX ---
    # Remove all punctuation (.,!?- etc.) from the entire string
    # This turns "cat." into "cat" and "dog." into "dog"
    s = s.translate(str.maketrans('', '', string.punctuation))
    # --- END FIX ---

    # Optional: collapse multiple spaces that might result from punctuation removal
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def jaro_winkler_match(s1: str, s2: str, threshold: float) -> bool:
    """
    Checks if Jaro-Winkler similarity is above a threshold.
    Ideal for short strings, codes, and labels.
    """
    return jellyfish.jaro_winkler_similarity(s1, s2) >= threshold


def jaccard_on_words_match(s1: str, s2: str, threshold: float) -> bool:
    """
    Checks if Jaccard similarity (on words) is above a threshold.
    Ideal for long paragraphs or descriptions.
    """
    words1 = set(s1.split())
    words2 = set(s2.split())

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    if union == 0:
        return 1.0 if intersection == 0 else 0.0  # Both empty or one empty

    score = intersection / union
    return score >= threshold

