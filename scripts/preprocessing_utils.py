# preprocessing_utils.py

import re
import spacy
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not present
nltk.download("stopwords")

# Load SpaCy POS tagger
nlp_pos = spacy.load("en_core_web_sm")

# TEXT CLEANING FUNCTION

def clean_text(text):
    """
    Lowercases text, removes unwanted characters, and filters tokens
    by POS (nouns and adjectives) and stopwords.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s/\.\-]", "", text)  # Retain clinical slashes/dashes

    stop_words = set(stopwords.words("english"))
    doc = nlp_pos(text)

    filtered = [
        token.text for token in doc
        if token.text not in stop_words and token.pos_ in ["NOUN", "ADJ"]
    ]

    return " ".join(filtered)


# NEGATIVE PHRASE REMOVAL FUNCTION
def remove_negative_phrases(text):
    """
    Removes sentences that contain negative indicators like 'no', 'not', etc.
    Useful before medical entity extraction to avoid false negatives.
    """
    negative_keywords = ["no", "not", "negative", "denying", "na"]
    sentences = re.split(r"(?<=\.)\s+", text)
    filtered = [
        sentence for sentence in sentences
        if not any(neg in sentence.lower() for neg in negative_keywords)
    ]
    return " ".join(filtered)


# TEXT CHUNKING UTILITY FUNCTION

def split_text_into_chunks(text, max_length=512):
    """
    Splits large input text into manageable chunks (by sentence),
    respecting token limit constraints for models like BERT.
    """
    sentences = re.split(r"(?<=\.)\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
