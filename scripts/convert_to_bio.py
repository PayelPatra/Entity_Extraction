import pandas as pd
import ast
import os
from nltk.tokenize import word_tokenize
from nltk import download

# Ensure punkt tokenizer is available
download('punkt')

def convert_char_spans_to_bio(text, spans):
    """
    Converts character-based entity spans into BIO format for tokens.
    """
    tokens = word_tokenize(text)
    bio_tags = []
    token_positions = []

    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        end = start + len(token)
        token_positions.append((token, start, end))
        cursor = end

    for token, start, end in token_positions:
        tag = "O"
        for ent_start, ent_end, ent_type in spans:
            if start == ent_start:
                tag = f"B-{ent_type}"
                break
            elif ent_start < start < ent_end:
                tag = f"I-{ent_type}"
                break
        bio_tags.append((token, tag))

    return bio_tags

def convert_csv_to_bio(input_csv, output_csv):
    """
    Reads annotated CSV with character spans and outputs token-level BIO CSV.
    """
    df = pd.read_csv(input_csv)
    sentence_id = 0

    with open(output_csv, "w", encoding="utf-8") as out_file:
        out_file.write("Sentence_ID,Token,Label\n")

        for _, row in df.iterrows():
            text = row["text"]
            try:
                spans = ast.literal_eval(row["labels"])
            except:
                spans = []

            tagged_tokens = convert_char_spans_to_bio(text, spans)

            for token, label in tagged_tokens:
                out_file.write(f"{sentence_id},{token},{label}\n")

            sentence_id += 1


    print(f"BIO-tagged output saved to: {output_path}")

if __name__ == "__main__":
    input_file = "input_annotations.csv"  # Replace or keep general for GitHub users

    print(f"BIO-tagged file saved at: {output_csv}")

if __name__ == "__main__":
    input_file = "input_annotations.csv"     # Update to your real file

    output_file = "bio_tagged_output.csv"
    convert_csv_to_bio(input_file, output_file)
