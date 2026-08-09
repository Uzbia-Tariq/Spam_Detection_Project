import streamlit as st
import joblib

from src.preprocessing import preprocess_text

# -----------------------
# Page Config
# -----------------------

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide"
)

# -----------------------
# Load CSS
# -----------------------

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------
# Load Model
# -----------------------

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# -----------------------
# Theme
# -----------------------

theme = st.sidebar.radio(
    "Theme",
    ["Dark", "Light"]
)

if theme == "Light":
    st.markdown("""
    <style>
    .stApp{
        background:#f5f7fb;
    }

    .title{
        color:#111;
    }

    .subtitle{
        color:#444;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------

st.markdown(
"""
<div class="title">
📩 SMS Spam Detector
</div>

<div class="subtitle">
Detect spam messages using Natural Language Processing and Machine Learning
</div>
""",
unsafe_allow_html=True
)

# -----------------------
# Layout
# -----------------------

left, right = st.columns([2.2,1])

# -----------------------
# LEFT
# -----------------------

with left:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    message = st.text_area(
        "Enter SMS",
        height=220,
        placeholder="Type or paste your SMS message..."
    )

    predict = st.button("Analyze Message")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# RIGHT
# -----------------------

with right:

    st.markdown("<h1>Model Overview</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
    <b>Algorithm</b><br><br>
    Multinomial Naive Bayes
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
    <b>Vectorizer</b><br><br>
    TF-IDF
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
    <b>Dataset</b><br><br>
    SMS Spam Collection
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
    <b>Accuracy</b><br><br>
    97%
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Prediction
# -----------------------

if predict:

    clean = preprocess_text(message)

    vector = vectorizer.transform([clean])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)

    confidence = probability.max() * 100

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 0:

        st.markdown(
        f"""
        <div class='result-safe'>
        <h2>✅ HAM (Safe Message)</h2>

        Confidence :
        <b>{confidence:.2f}%</b>
        </div>
        """,
        unsafe_allow_html=True
        )

    else:

        st.markdown(
        f"""
        <div class='result-spam'>
        <h2>🚨 SPAM Message</h2>

        Confidence :
        <b>{confidence:.2f}%</b>
        </div>
        """,
        unsafe_allow_html=True
        )

    st.progress(confidence/100)