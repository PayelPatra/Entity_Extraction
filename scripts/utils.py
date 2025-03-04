# Function to manually calculate Precision, Recall, F1 Score, and Accuracy
def evaluate_performance(true_labels, predicted_labels):
    true_positive = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p == 1)
    false_positive = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1)
    false_negative = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)
    true_negative = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p == 0)

    # Precision = TP / (TP + FP)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

    # Recall = TP / (TP + FN)
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1_score, 4),
        "Accuracy": round(accuracy, 4)
    }
