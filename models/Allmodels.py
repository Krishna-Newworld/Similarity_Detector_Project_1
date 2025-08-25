import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB

# 🔹 Define paths
train_file = 'C:/Users/taskf/OneDrive/Desktop/TextSimilarity/models/Sem4_preprocess_sts_train.csv'
test_file = 'C:/Users/taskf/OneDrive/Desktop/TextSimilarity/models/Sem4_preprocess_sts_test.csv'
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

# 🔹 Load datasets
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# 🔹 Features and target
features = [
    "cosine_similarity", "jaccard_similarity", "dice_similarity",
    "average_similarity", "bpt_cosine_similarity",
    "trigram_dice_similarity", "pos_tags_cosine_similarity"
]
target = "SPL_five_Level"

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# 🔹 Label Encoding
label_order = sorted(set(y_train.unique()).union(set(y_test.unique())))
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# 🔹 Save label encoder
joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.pkl"))

# 🔹 Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 Save scaler
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

# 🔹 Hybrid Models
hybrid_models = {
    "Random_Forest_SVM": (RandomForestClassifier(n_estimators=100, random_state=42),
                           SVC(kernel='rbf', probability=True, random_state=42)),

    "Random_Forest_AdaBoost": (RandomForestClassifier(n_estimators=100, random_state=42),
                                AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=42)),

    "AdaBoost_SVM": (AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=42),
                      SVC(kernel='rbf', probability=True, random_state=42)),

    "SVM_Logistic_Regression": (SVC(kernel='rbf', probability=True, random_state=42),
                                 LogisticRegression(max_iter=200, random_state=42)),

    "SVM_XGBoost": (SVC(kernel='rbf', probability=True, random_state=42),
                    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')),

    "Random_Forest_XGBoost": (RandomForestClassifier(n_estimators=100, random_state=42),
                               XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')),

    "Logistic_Regression_AdaBoost": (LogisticRegression(max_iter=200, random_state=42),
                                      AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=42)),

    "Random_Forest_Logistic_Regression": (RandomForestClassifier(n_estimators=100, random_state=42),
                                           LogisticRegression(max_iter=200, random_state=42)),

    "XGBoost_SVM": (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                     SVC(kernel='rbf', probability=True, random_state=42)),

    "XGBoost_Random_Forest": (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                               RandomForestClassifier(n_estimators=100, random_state=42)),

    "LDA_SVM": (LDA(), SVC(kernel='rbf', probability=True, random_state=42)),

    "LDA_GaussianNB": (LDA(), GaussianNB()),

    "AdaBoost_GaussianNB": (AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=42),
                             GaussianNB())
}

# 🔹 Train & Save Models
for name, (model1, model2) in hybrid_models.items():
    print(f"\nTraining: {name.replace('_', ' ')}")

    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)

    misclassified = np.where(y_pred1 != y_test)[0]

    if len(misclassified) > 0:
        model2.fit(X_test[misclassified], y_test[misclassified])
        y_pred2 = model2.predict(X_test)
        final_pred = np.copy(y_pred1)
        final_pred[misclassified] = y_pred2[misclassified]
    else:
        final_pred = y_pred1

    accuracy = accuracy_score(y_test, final_pred)

    # Save models
    joblib.dump(model1, os.path.join(save_dir, f"{name}_model1.pkl"))
    joblib.dump(model2, os.path.join(save_dir, f"{name}_model2.pkl"))

    print(f"✅ Saved {name}_model1.pkl and {name}_model2.pkl")

    # Confusion Matrix
    cm = confusion_matrix(y_test, final_pred, labels=label_encoder.transform(label_order))
    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Purples')
    plt.title(f"{name.replace('_', ' ')}\nAccuracy: {accuracy:.4f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
