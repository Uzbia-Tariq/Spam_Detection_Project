import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from preprocessing import preprocess_text

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/spam_sms.csv", encoding="latin-1")

df.rename(columns={
    "v1": "label",
    "v2": "message"
}, inplace=True)

df = df[["label", "message"]]

# ==========================================
# Preprocess Text
# ==========================================

df["clean_text"] = df["message"].apply(preprocess_text)

# Convert labels

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# ==========================================
# TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(df["clean_text"])

y = df["label"]

# ==========================================
# Train Model
# ==========================================

model = MultinomialNB()

model.fit(X, y)

# ==========================================
# Create Models Folder
# ==========================================

os.makedirs("models", exist_ok=True)

# ==========================================
# Save Model
# ==========================================

with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

# ==========================================
# Save Vectorizer
# ==========================================

with open("models/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("=" * 50)
print("✅ Model Saved Successfully")
print("✅ Vectorizer Saved Successfully")
print("=" * 50)