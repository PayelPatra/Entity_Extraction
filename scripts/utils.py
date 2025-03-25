# utils.py

from sklearn.metrics import classification_report

def evaluate_performance(true_labels, predicted_labels):
    
    report = classification_report(
        true_labels,
        predicted_labels,
        output_dict=True,
        zero_division=0
    )

    # Exclude 'O' label from macro avg
    entity_labels = [label for label in report if label not in ["accuracy", "macro avg", "weighted avg", "O"]]

    precision = sum(report[label]["precision"] for label in entity_labels) / len(entity_labels)
    recall = sum(report[label]["recall"] for label in entity_labels) / len(entity_labels)
    f1 = sum(report[label]["f1-score"] for label in entity_labels) / len(entity_labels)
    accuracy = report["accuracy"]

    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Accuracy": round(accuracy, 4)
    }
