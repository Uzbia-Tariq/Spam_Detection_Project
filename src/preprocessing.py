import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Tokenization
    words = word_tokenize(text)

    # 3. Remove punctuation
    words = [word for word in words if word not in string.punctuation]

    # 4. Remove stopwords
    words = [word for word in words if word not in stop_words]

    # 5. Stemming
    words = [ps.stem(word) for word in words]

    # Join words again
    return " ".join(words)