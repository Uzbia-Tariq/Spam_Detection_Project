import streamlit as st
import joblib
from src.preprocessing import preprocess_text

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Spam Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME STATE
# =========================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# =========================================================
# CSS
# =========================================================

with open("style.css", "r", encoding="utf-8") as f:
    css = f.read()

theme_css = """
:root {
    --bg: #080d18;
    --sidebar: #091426;
    --panel: #101b30;
    --panel-2: #13233f;
    --input: #0d192d;
    --border: rgba(148,163,184,.16);
    --text: #f5f7ff;
    --muted: #91a4c4;
}

body {
    background: var(--bg);
}
"""

if not st.session_state.dark_mode:
    theme_css = """
    :root {
        --bg: #f4f7fb;
        --sidebar: #ffffff;
        --panel: #ffffff;
        --panel-2: #eef3fb;
        --input: #f8fafc;
        --border: rgba(71,85,105,.18);
        --text: #172033;
        --muted: #64748b;
    }

    body {
        background: var(--bg);
    }
    """

st.html(f"<style>{css}{theme_css}</style>")

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# =========================================================
# NAVIGATION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">SG</div>
            <div>
                <div class="brand-title">Spam Guard</div>
                <div class="brand-subtitle">AI SMS Detection</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-heading">Navigation</div>', unsafe_allow_html=True)

    if st.button("Home", key="nav_home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()

    if st.button("Dashboard", key="nav_dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"
        st.rerun()

    if st.button("Model Details", key="nav_model", use_container_width=True):
        st.session_state.page = "Model Details"
        st.rerun()

    if st.button("About Project", key="nav_about", use_container_width=True):
        st.session_state.page = "About Project"
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Real theme control. No red Streamlit switch is used.
    toggle_label = "Dark Mode" if st.session_state.dark_mode else "Light Mode"

    if st.button(
        toggle_label,
        key="theme_button",
        use_container_width=True,
    ):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown(
        f"""
        <div class="theme-status">
            <span class="theme-dot"></span>
            {"Dark appearance" if st.session_state.dark_mode else "Light appearance"}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="version">Spam Guard v2.0</div>', unsafe_allow_html=True)

# =========================================================
# DASHBOARD
# =========================================================

if st.session_state.page == "Dashboard":

    st.markdown(
        """
        <div class="page-header">
            <div class="eyebrow">SPAM GUARD</div>
            <h1>Dashboard</h1>
            <p>Overview of the SMS spam detection system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:
        st.markdown('<div class="stat-card"><small>ALGORITHM</small><strong>Naive Bayes</strong></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="stat-card"><small>VECTORIZER</small><strong>TF-IDF</strong></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="stat-card"><small>DATASET</small><strong>SMS Spam</strong></div>', unsafe_allow_html=True)
    with d:
        st.markdown('<div class="stat-card"><small>ACCURACY</small><strong>97%</strong></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="large-card">
            <h3>Detection Pipeline</h3>
            <div class="pipeline">
                <span>SMS Input</span><b>→</b>
                <span>Preprocessing</span><b>→</b>
                <span>TF-IDF</span><b>→</b>
                <span>Naive Bayes</span><b>→</b>
                <span>Result</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# MODEL DETAILS
# =========================================================

elif st.session_state.page == "Model Details":

    st.markdown(
        """
        <div class="page-header">
            <div class="eyebrow">SPAM GUARD</div>
            <h1>Model Details</h1>
            <p>Technical information about the trained model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="large-card">
                <div class="card-kicker">CLASSIFIER</div>
                <h2>Multinomial Naive Bayes</h2>
                <p>Used for text classification and SMS spam detection.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="large-card">
                <div class="card-kicker">FEATURE EXTRACTION</div>
                <h2>TF-IDF</h2>
                <p>Converts message text into numerical features for classification.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# ABOUT
# =========================================================

elif st.session_state.page == "About Project":

    st.markdown(
        """
        <div class="page-header">
            <div class="eyebrow">SPAM GUARD</div>
            <h1>About Project</h1>
            <p>Machine-learning based SMS spam detection.</p>
        </div>

        <div class="large-card">
            <div class="card-kicker">PROJECT PURPOSE</div>
            <h2>Intelligent SMS Security</h2>
            <p>
                Spam Guard analyzes SMS messages and predicts whether a
                message is unwanted spam or a normal message. The system
                uses natural language preprocessing, TF-IDF vectorization
                and Multinomial Naive Bayes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# HOME
# =========================================================

else:

    # Header is deliberately separate from the hero so nothing is hidden.
    st.markdown(
        """
        <div class="home-header">
            <div>
                <div class="eyebrow">INTELLIGENT MESSAGE SECURITY</div>
                <h1>SMS Spam Detector</h1>
            </div>
            <div class="online-badge"><i></i> MODEL ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Full-width hero. The old right-side phone box has been removed.
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">AI POWERED DETECTION</div>
            <h2>Protect your inbox from <span>spam.</span></h2>
            <p>
                Analyze SMS messages using Natural Language Processing,
                TF-IDF Vectorization and Multinomial Naive Bayes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-title">
            <div>
                <h2>Analyze SMS</h2>
                <p>Enter a message below and run the detector.</p>
            </div>
            <div class="number-tag">01</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_message = st.text_area(
        "SMS message",
        height=105,
        placeholder="Paste or type an SMS message here...",
        label_visibility="collapsed",
    )

    # Compact action row — READY indicator removed.
    col1, col2 = st.columns([1.8, 1.0])

    with col1:
        detect = st.button("Detect Spam", key="detect", use_container_width=True)

    with col2:
        clear = st.button("Clear", key="clear", use_container_width=True)

    if clear:
        st.rerun()

    # Model overview BELOW the input, not in a distracting top-right box.
    st.markdown(
        """
        <div class="section-title model-section">
            <div>
                <h2>Model Overview</h2>
                <p>Current detection engine.</p>
            </div>
            <div class="number-tag">02</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    o1, o2, o3, o4 = st.columns(4)

    with o1:
        st.markdown('<div class="overview-card"><small>ALGORITHM</small><strong>Naive Bayes</strong></div>', unsafe_allow_html=True)
    with o2:
        st.markdown('<div class="overview-card"><small>VECTORIZER</small><strong>TF-IDF</strong></div>', unsafe_allow_html=True)
    with o3:
        st.markdown('<div class="overview-card"><small>DATASET</small><strong>SMS Spam</strong></div>', unsafe_allow_html=True)
    with o4:
        st.markdown('<div class="overview-card"><small>ACCURACY</small><strong>97%</strong></div>', unsafe_allow_html=True)

    # =====================================================
    # PREDICTION
    # =====================================================

    if detect:

        if not user_message.strip():
            st.warning("Please enter an SMS message.")

        else:
            clean_text = preprocess_text(user_message)
            vector = vectorizer.transform([clean_text])

            prediction = model.predict(vector)[0]
            probabilities = model.predict_proba(vector)[0]

            ham = probabilities[0] * 100
            spam = probabilities[1] * 100
            confidence = max(ham, spam)

            # "NORMAL MESSAGE" is used instead of "SAFE MESSAGE".
            if prediction == 1:
                title = "SPAM MESSAGE"
                description = "This message is likely unwanted or promotional spam."
                result_class = "spam-result"
            else:
                title = "NORMAL MESSAGE"
                description = "This message does not show significant spam indicators."
                result_class = "normal-result"

            st.markdown(
                f"""
                <div class="result {result_class}">
                    <div>
                        <small>PREDICTION</small>
                        <h2>{title}</h2>
                        <p>{description}</p>
                    </div>
                    <div class="confidence">
                        <small>CONFIDENCE</small>
                        <strong>{confidence:.1f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            p1, p2 = st.columns(2)

            with p1:
                st.markdown(
                    f'<div class="prob-line"><span>Normal</span><b>{ham:.1f}%</b></div>',
                    unsafe_allow_html=True,
                )
                st.progress(ham / 100)

            with p2:
                st.markdown(
                    f'<div class="prob-line"><span>Spam</span><b>{spam:.1f}%</b></div>',
                    unsafe_allow_html=True,
                )
                st.progress(spam / 100)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Characters", len(user_message))
            with c2:
                st.metric("Words", len(user_message.split()))
            with c3:
                st.metric("Confidence", f"{confidence:.1f}%")

    st.markdown(
        '<div class="footer">Spam Guard • NLP • TF-IDF • Multinomial Naive Bayes • Streamlit</div>',
        unsafe_allow_html=True,
    )
