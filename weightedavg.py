# weightavg.py

# label → numeric midpoint
label_to_score = {
    "NON": 10,
    "MIN": 30,
    "LOW": 50,
    "MID": 70,
    "HIGH": 90
}

# model weights (average of SICK & STS accuracies)
model_weights = {
    "AdaBoost_GaussianNB": (76 + 66) / 2,
    "AdaBoost_SVM": (84 + 71) / 2,
    "LDA_GaussianNB": (78 + 66) / 2,
    "LDA_SVM": (84 + 71) / 2,
    "Logistic_Regression_AdaBoost": (81 + 67) / 2,
    "Random_Forest_AdaBoost": (71 + 61) / 2,
    "Random_Forest_Logistic_Regression": (71 + 59) / 2,
    "Random_Forest_SVM": (73 + 63) / 2,
    "Random_Forest_XGBoost": (97 + 99) / 2,
    "SVM_Logistic_Regression": (79 + 66) / 2,
    "SVM_XGBoost": (99 + 99) / 2,
    "XGBoost_Random_Forest": (99 + 99) / 2,
    "XGBoost_SVM": (75 + 66) / 2,
}


def compute_weighted_score(predictions: dict) -> dict:
    """
    Compute final weighted plagiarism score based on model predictions.
    
    Args:
        predictions (dict): model_name -> label (e.g., {"SVM_XGBoost": "HIGH"})
    
    Returns:
        dict: {
            "plagiarism_percent": float,
            "final_label": str
        }
    """
    weighted_sum = 0
    total_weight = 0

    for model, label in predictions.items():
        if model not in model_weights or label not in label_to_score:
            continue  # skip if model or label is unrecognized
        score = label_to_score[label]
        weight = model_weights[model]
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return {"plagiarism_percent": None, "final_label": "UNKNOWN"}

    final_score = weighted_sum / total_weight

    # map score back to label
    if 0 <= final_score <= 20:
        label = "NON"
    elif 21 <= final_score <= 40:
        label = "MIN"
    elif 41 <= final_score <= 60:
        label = "LOW"
    elif 61 <= final_score <= 80:
        label = "MID"
    else:
        label = "HIGH"

    return {
        "plagiarism_percent": round(final_score, 2),
        "final_label": label
    }
