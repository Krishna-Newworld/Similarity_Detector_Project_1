import os
import joblib
import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize, pos_tag
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------- NLTK Setup ------------------- #
# Point to local nltk_data/ (either inside models/ or project root)
NLTK_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(NLTK_PATH):
    NLTK_PATH = os.path.join(os.path.dirname(__file__), "..", "nltk_data")

nltk.data.path.append(NLTK_PATH)

# ---------------- Feature Computation ---------------- #

def cosine_from_texts(text1, text2):
    vec = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]

def jaccard_similarity(set1, set2):
    set1, set2 = set(set1), set(set2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0.0

def dice_similarity(set1, set2):
    set1, set2 = set(set1), set(set2)
    overlap = len(set1 & set2)
    return 2 * overlap / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0.0

def trigram_dice(sent1, sent2):
    trigrams1 = list(ngrams(word_tokenize(sent1.lower()), 3))
    trigrams2 = list(ngrams(word_tokenize(sent2.lower()), 3))
    return dice_similarity(trigrams1, trigrams2)

def pos_tags_cosine_similarity(sent1, sent2):
    pos1 = ' '.join([tag for _, tag in pos_tag(word_tokenize(sent1))])
    pos2 = ' '.join([tag for _, tag in pos_tag(word_tokenize(sent2))])
    return cosine_from_texts(pos1, pos2)

def extract_features_from_sentence_pair(sent1, sent2):
    tokens1 = word_tokenize(sent1.lower())
    tokens2 = word_tokenize(sent2.lower())

    cosine_sim = cosine_from_texts(sent1, sent2)
    jaccard_sim = jaccard_similarity(tokens1, tokens2)
    dice_sim = dice_similarity(tokens1, tokens2)
    avg_sim = (cosine_sim + jaccard_sim + dice_sim) / 3
    bpt_cosine = cosine_from_texts(sent1[::-1], sent2[::-1])  # Placeholder for BPT
    trigram_dice_sim = trigram_dice(sent1, sent2)
    pos_cos_sim = pos_tags_cosine_similarity(sent1, sent2)

    return pd.DataFrame([{
        "cosine_similarity": cosine_sim,
        "jaccard_similarity": jaccard_sim,
        "dice_similarity": dice_sim,
        "average_similarity": avg_sim,
        "bpt_cosine_similarity": bpt_cosine,
        "trigram_dice_similarity": trigram_dice_sim,
        "pos_tags_cosine_similarity": pos_cos_sim
    }])

# ---------------- Load Preprocessing Objects ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "..", "saved_models")

label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

if not os.path.exists(scaler_path) or not os.path.exists(label_encoder_path):
    raise FileNotFoundError("🚫 Missing scaler.pkl or label_encoder.pkl. Make sure saved_models/ is pushed to GitHub.")

scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# ---------------- Predict Function ---------------- #

def predict_with_all_models(sent1, sent2):
    features_df = extract_features_from_sentence_pair(sent1, sent2)
    scaled_features = scaler.transform(features_df)

    predictions = {}
    for filename in os.listdir(model_dir):
        if filename.endswith("_model1.pkl"):
            model_name = filename.replace("_model1.pkl", "").replace("__", " ")
            model1_path = os.path.join(model_dir, filename)
            model2_path = model1_path.replace("_model1.pkl", "_model2.pkl")

            model1 = joblib.load(model1_path)

            # Model2 is optional
            model2 = joblib.load(model2_path) if os.path.exists(model2_path) else None

            pred1 = model1.predict(scaled_features)
            pred2 = model2.predict(scaled_features) if model2 else None

            final_pred = pred2[0] if pred2 is not None else pred1[0]
            label = label_encoder.inverse_transform([final_pred])[0]

            predictions[model_name] = label

    return predictions
