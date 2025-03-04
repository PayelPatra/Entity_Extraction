import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import evaluate_performance  # Importing manual evaluation function

# Define Paths for All Models & Corresponding Test Data
models = {
    "BioBERT": {
        "model_path": "./models/biobert",
        "test_data": "./output/filtered_extracted_entities_with_biobert.csv"
    },
    "BioClinicalBERT": {
        "model_path": "./models/bioclinicalbert",
        "test_data": "./output/filtered_extracted_entities_with_bioclinicalbert.csv"
    },
    "BlueBERT": {
        "model_path": "./models/bluebert",
        "test_data": "./output/filtered_extracted_entities_with_bluebert.csv"
    },
    "RoBERTa": {
        "model_path": "./models/roberta",
        "test_data": "./output/filtered_extracted_entities_with_roberta.csv"
    },
    "PubMedBERT": {
        "model_path": "./models/pubmedbert",
        "test_data": "./output/filtered_extracted_entities_with_pubmedbert.csv"
    }
}

# Define Label Columns (Same for All Models)
label_columns = ["age", "gender", "disease", "symptoms", "medication", "dose", "cancer stage"]

# Iterate Through Each Model and Evaluate Performance
for model_name, paths in models.items():
    print(f"\n==== Evaluating {model_name} ====\n")

    # Load the Test Dataset
    test_data_path = paths["test_data"]
    if not os.path.exists(test_data_path):
        print(f"⚠ Warning: Test data not found for {model_name}. Skipping...")
        continue

    test_df = pd.read_csv(test_data_path)
    print(f"Test dataset for {model_name} loaded with {len(test_df)} samples.")

    # Normalize column names
    test_df.columns = test_df.columns.str.strip().str.lower()
    
    # Combine all columns into a single text input
    test_df['combined_text'] = test_df.fillna('').apply(lambda row: ' '.join(row.astype(str)), axis=1)

    # Extract True Labels
    true_labels = {col: test_df[col].apply(lambda x: set(str(x).split()) if isinstance(x, str) else set()).tolist() for col in label_columns}

    # Load Tokenizer & Trained Model
    model_path = paths["model_path"]
    if not os.path.exists(model_path):
        print(f"⚠ Warning: Model not found for {model_name}. Skipping...")
        continue

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Tokenize the Test Data
    def encode_data(tokenizer, texts, max_length=128):
        encoded = tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            padding="longest",
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded["input_ids"], encoded["attention_mask"]

    test_input_ids, test_attention_mask = encode_data(tokenizer, test_df['combined_text'].tolist(), max_length=128)

    # Get Model Predictions
    model.eval()
    with torch.no_grad():
        outputs = model(test_input_ids, attention_mask=test_attention_mask)
        predictions = torch.sigmoid(outputs.logits).cpu().numpy()  # Multi-label probabilities

    # Adjust Prediction Threshold
    threshold = 0.48
    predicted_labels = []
    for i, row in test_df.iterrows():
        predicted_entities = {}
        for col_idx, col in enumerate(label_columns):
            if predictions[i, col_idx] > threshold:
                predicted_entities[col] = set(str(row[col]).split())  # Use actual content of the column
        predicted_labels.append(predicted_entities)

    # Calculate Performance Metrics
    total_precision, total_recall, total_f1, total_accuracy = 0, 0, 0, 0
    total_count = len(label_columns)

    for col in label_columns:
        true_col = true_labels[col]
        pred_col = [predicted[col] if col in predicted else set() for predicted in predicted_labels]

        true_positive = sum(len(tc & pc) for tc, pc in zip(true_col, pred_col))
        false_positive = sum(len(pc - tc) for tc, pc in zip(true_col, pred_col))
        false_negative = sum(len(tc - pc) for tc, pc in zip(true_col, pred_col))

        # Precision, Recall, F1
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Accuracy
        total_entities = sum(len(tc) for tc in true_col)
        correct_entities = sum(len(tc & pc) for tc, pc in zip(true_col, pred_col))
        accuracy = correct_entities / total_entities if total_entities > 0 else 0

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_accuracy += accuracy

    #  Average Metrics for the Model
    avg_precision = total_precision / total_count
    avg_recall = total_recall / total_count
    avg_f1 = total_f1 / total_count
    avg_accuracy = total_accuracy / total_count

    print(f" Overall Results for {model_name}:")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"Accuracy: {avg_accuracy:.4f}")
 
