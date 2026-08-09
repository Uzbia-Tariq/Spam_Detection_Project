import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocessing import preprocess_text

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/spam_sms.csv", encoding="latin-1")

# Rename Columns
df.rename(columns={
    "v1": "label",
    "v2": "message"
}, inplace=True)

# Keep only required columns
df = df[["label", "message"]]

# ==========================================
# Text Preprocessing
# ==========================================

df["clean_text"] = df["message"].apply(preprocess_text)

# ==========================================
# Convert Labels
# ham = 0
# spam = 1
# ==========================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# ==========================================
# TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("=" * 50)
print("Training Data Shape :", X_train.shape)
print("Testing Data Shape  :", X_test.shape)

# ==========================================
# Train Naive Bayes Model
# ==========================================

model = MultinomialNB()

model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully!")

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Model Evaluation
# ==========================================

print("\n" + "=" * 50)
print("Accuracy Score")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("\n" + "=" * 50)
print("Confusion Matrix")
print("=" * 50)

print(confusion_matrix(y_test, y_pred))

print("\n" + "=" * 50)
print("Classification Report")
print("=" * 50)

print(classification_report(y_test, y_pred))