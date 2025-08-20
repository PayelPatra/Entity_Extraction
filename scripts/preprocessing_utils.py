import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
_STOP = set(stopwords.words("english"))

_SPACY = None
def _spacy():
    global _SPACY
    if _SPACY is None:
        import spacy
        _SPACY = spacy.load("en_core_web_sm")
    return _SPACY

def clean_text(text, mode: str = "minimal"):
    if not isinstance(text, str):
        text = str(text)
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"[^\w\s/\.\-]", "", t)
    if mode != "viz":
        return t
    nlp = _spacy()
    doc = nlp(t)
    kept = [tok.text for tok in doc if tok.text.lower() not in _STOP and tok.pos_ in ("NOUN", "ADJ")]
    return " ".join(kept)

def remove_negative_phrases(text, enable: bool = False):
    if not enable:
        return text
    negatives = (" no ", " not ", " negative ", " denying ", " na ")
    sentences = re.split(r"(?<=\.)\s+", text)
    kept = [s for s in sentences if not any(neg in f" {s.lower()} " for neg in negatives)]
    return " ".join(kept)

def split_text_into_chunks(text, max_length: int = 512):
    sentences = re.split(r"(?<=\.)\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_length:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks
