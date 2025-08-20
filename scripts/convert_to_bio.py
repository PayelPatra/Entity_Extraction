import os
import ast
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import download

download("punkt")

def _parse_spans(raw):
    if isinstance(raw, list):
        spans = raw
    else:
        s = str(raw).strip()
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = ast.literal_eval(s)
        spans = parsed
    norm = []
    for item in spans if spans is not None else []:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            start, end, label = item[0], item[1], item[2]
            norm.append((int(start), int(end), str(label)))
        elif isinstance(item, dict):
            start = int(item.get("start", item.get("begin", 0)))
            end   = int(item.get("end",   item.get("stop", 0)))
            label = str(item.get("label", item.get("type", "O")))
            norm.append((start, end, label))
    return norm

def convert_char_spans_to_bio(text, spans):
    tokens = word_tokenize(text)
    bio_tags = []
    positions = []
    cursor = 0
    for tok in tokens:
        start = text.find(tok, cursor)
        end = start + len(tok)
        positions.append((start, end))
        cursor = end
    for (start, end) in positions:
        tag = "O"
        for (s, e, lab) in spans:
            if start == s and start < e:
                tag = f"B-{lab}"
                break
            elif s < start < e:
                tag = f"I-{lab}"
                break
        bio_tags.append(tag)
    return tokens, bio_tags, positions

def write_token_level_csv(df_in, out_path_tokens):
    rows = []
    for sid, row in df_in.iterrows():
        text = row["text"]
        spans = _parse_spans(row["labels"])
        tokens, tags, _ = convert_char_spans_to_bio(text, spans)
        for i, (tok, tag) in enumerate(zip(tokens, tags)):
            rows.append({
                "Sentence_ID": sid,
                "Token_Index": i,
                "Token": tok,
                "Label": tag
            })
    pd.DataFrame(rows).to_csv(out_path_tokens, index=False)

def write_sequence_level_csv(df_in, out_path_seq):
    rows = []
    for _, row in df_in.iterrows():
        text = row["text"]
        spans = _parse_spans(row["labels"])
        _, tags, _ = convert_char_spans_to_bio(text, spans)
        rows.append({"text": text, "bio_labels": " ".join(tags)})
    pd.DataFrame(rows).to_csv(out_path_seq, index=False)

def convert_csv_to_bio(input_csv, out_seq_csv, out_token_csv):
    df = pd.read_csv(input_csv)
    if "text" not in df.columns or "labels" not in df.columns:
        raise ValueError("Input CSV must contain columns: 'text' and 'labels'.")
    write_sequence_level_csv(df, out_seq_csv)
    write_token_level_csv(df, out_token_csv)
    print(f"[OK] Sequence-level BIO saved to: {out_seq_csv}")
    print(f"[OK] Token-level BIO saved to:    {out_token_csv}")

if __name__ == "__main__":
    input_file = "input_annotations.csv"
    out_seq = "bio_tagged_output.csv"
    out_tok = "bio_tagged_output_tokens.csv"
    convert_csv_to_bio(input_file, out_seq, out_tok)
