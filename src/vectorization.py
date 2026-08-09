import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess_text

# Load Dataset
df = pd.read_csv("data/spam_sms.csv", encoding="latin-1")

df = df.rename(columns={
    "v1": "label",
    "v2": "message"
})

# Clean Text
df["clean_text"] = df["message"].apply(preprocess_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df["clean_text"])

print("Shape of TF-IDF Matrix:")
print(X.shape)