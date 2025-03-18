# preprocessing_utils.py
import re
import spacy
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load SpaCy model
nlp_pos = spacy.load('en_core_web_sm')

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s/\.\-]", "", text)

    stop_words = set(stopwords.words('english'))
    doc = nlp_pos(text)
    cleaned_tokens = [token.text for token in doc if token.text not in stop_words and token.pos_ in ['NOUN', 'ADJ']]

    return " ".join(cleaned_tokens)

# Remove negated or negative sentences
def remove_negative_phrases(text):
    negative_indicators = ['no', 'not', 'negative', 'denying', 'na']
    sentences = re.split(r'(?<=\.)\s+', text)
    filtered_sentences = [sentence for sentence in sentences if not any(neg in sentence.lower() for neg in negative_indicators)]
    return ' '.join(filtered_sentences)

# Split text into chunks
def split_text_into_chunks(text, max_length=512):
    sentences = re.split(r'(?<=\.)\s+', text)
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
