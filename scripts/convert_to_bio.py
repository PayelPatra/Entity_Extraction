import pandas as pd
import ast
import os
from nltk.tokenize import word_tokenize
from nltk import download

download('punkt')  # For word tokenization

def char_level_to_token_bio(text, entities):
    """
    Converts character-level entity spans into token-level BIO tags.
    """
    tokens = word_tokenize(text)
    token_spans = []
    current = 0

    for token in tokens:
        start = text.find(token, current)
        end = start + len(token)
        token_spans.append((token, start, end))
        current = end

    bio_labels = []
    for token, start, end in token_spans:
        label = "O"
        for ent_start, ent_end, ent_label in entities:
            if start == ent_start:
                label = f"B-{ent_label}"
                break
            elif start > ent_start and end <= ent_end:
                label = f"I-{ent_label}"
                break
        bio_labels.append((token, label))

    return bio_labels

def convert_doccano_csv_to_bio(input_path, output_path):
    df = pd.read_csv(input_path)
    sentence_id = 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Sentence_ID,Token,Label\n")

        for _, row in df.iterrows():
            text = row['text']
            try:
                entities = ast.literal_eval(row['labels'])
            except:
                entities = []
            
            bio_tagged = char_level_to_token_bio(text, entities)

            for token, label in bio_tagged:
                f.write(f"{sentence_id},{token},{label}\n")
            
            sentence_id += 1

    print(f"BIO-tagged output saved to: {output_path}")

if __name__ == "__main__":
    input_file = "input_annotations.csv"  # Replace or keep general for GitHub users
    output_file = "bio_tagged_output.csv"

    convert_doccano_csv_to_bio(input_file, output_file)
