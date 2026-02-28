# import streamlit as st
# import joblib
# import re
# import os
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# import PyPDF2

# # -------------------------
# # NLTK DOWNLOAD (RUNS ONLY ONCE)
# # -------------------------
# @st.cache_resource(show_spinner=False)
# def download_nltk():
#     nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
#     os.makedirs(nltk_data_path, exist_ok=True)
#     nltk.data.path.append(nltk_data_path)

#     packages = ["punkt_tab", "punkt", "stopwords", "wordnet", "omw-1.4"]
#     for pkg in packages:
#         try:
#             nltk.data.find(pkg)
#         except LookupError:
#             nltk.download(pkg, download_dir=nltk_data_path)

# download_nltk()

# # -------------------------
# # Page Config
# # -------------------------
# st.set_page_config(page_title="Contract Risk Analyzer", page_icon="📄", layout="wide")

# # -------------------------
# # Modern CSS
# # -------------------------
# st.markdown("""
# <style>
# .block-container {padding-top: 2rem;}
# .hero {
#     background: linear-gradient(135deg,#1f4e79,#4f8cc9);
#     padding: 35px;
#     border-radius: 18px;
#     color: white;
#     text-align: center;
#     margin-bottom: 25px;
# }
# .card {
#     background: white;
#     padding: 25px;
#     border-radius: 18px;
#     box-shadow: 0 8px 24px rgba(0,0,0,0.08);
# }
# .risk {
#     font-size: 28px;
#     font-weight: 700;
#     padding: 18px;
#     border-radius: 14px;
#     text-align: center;
# }
# .low {background:#d4edda;color:#155724;}
# .medium {background:#fff3cd;color:#856404;}
# .high {background:#f8d7da;color:#721c24;}
# .footer {text-align:center;color:gray;margin-top:40px}
# </style>
# """, unsafe_allow_html=True)

# # -------------------------
# # Load Models
# # -------------------------
# @st.cache_resource
# def load_models():
#     model = joblib.load("risk_model.pkl")
#     vectorizer = joblib.load("tfidf_vectorizer.pkl")
#     label_encoder = joblib.load("label_encoder.pkl")
#     return model, vectorizer, label_encoder

# model, vectorizer, le = load_models()

# # -------------------------
# # NLP Setup
# # -------------------------
# stop_words = set(stopwords.words('english'))
# legal_keep = {"shall", "not", "may", "must"}
# stop_words = stop_words - legal_keep
# lemmatizer = WordNetLemmatizer()

# # -------------------------
# # Preprocess
# # -------------------------
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     tokens = word_tokenize(text)
#     tokens = [word for word in tokens if word not in stop_words]
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     return " ".join(tokens)

# # -------------------------
# # HERO
# # -------------------------
# st.markdown('<div class="hero"><h1>📄 Intelligent Contract Risk Analyzer</h1><p>AI-powered clause level legal risk detection</p></div>', unsafe_allow_html=True)

# # -------------------------
# # INPUT SECTION
# # -------------------------
# container = st.container()

# with container:
#     st.markdown('<div class="card">', unsafe_allow_html=True)

#     input_method = st.radio("Choose Input Method:", ["Paste Text", "Upload PDF"])

#     user_input = ""

#     if input_method == "Paste Text":
#         user_input = st.text_area(
#             "Paste legal clause here",
#             height=220,
#             placeholder="Example: The vendor shall not terminate the agreement without prior written notice..."
#         )

#     elif input_method == "Upload PDF":
#         uploaded_file = st.file_uploader("Upload Contract PDF", type=["pdf"])

#         if uploaded_file is not None:
#             pdf_reader = PyPDF2.PdfReader(uploaded_file)
#             extracted_text = ""

#             for page in pdf_reader.pages:
#                 text = page.extract_text()
#                 if text:
#                     extracted_text += text

#             user_input = extracted_text
#             st.success("PDF uploaded and text extracted successfully!")
#             st.text_area("Extracted Text Preview:", user_input[:1000], height=150)

#     analyze = st.button("🔍 Analyze Risk", use_container_width=True)
#     st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

#     # -------------------------
#     # RESULT SECTION
#     # -------------------------
#     # st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Risk Assessment")

#     if analyze:
#         if user_input.strip() == "":
#             st.warning("Please enter a contract clause.")
#         else:
#             with st.spinner("Analyzing legal risk..."):
#                 cleaned = preprocess_text(user_input)
#                 vector = vectorizer.transform([cleaned])
#                 prediction = model.predict(vector)
#                 probabilities = model.predict_proba(vector)

#                 risk_label = le.inverse_transform(prediction)[0]
#                 confidence = round(max(probabilities[0]) * 100, 2)

#             css_class = risk_label.lower()
#             st.markdown(
#                 f'<div class="risk {css_class}">{risk_label.upper()} RISK<br><span style="font-size:18px">Confidence: {confidence}%</span></div>',
#                 unsafe_allow_html=True
#             )

#             st.progress(int(confidence))
#     else:
#         st.info("Result will appear here after analysis")

#     st.markdown('</div>', unsafe_allow_html=True)

# # Footer
# st.markdown('<div class="footer">Built using Streamlit • Machine Learning • NLP</div>', unsafe_allow_html=True)


import streamlit as st
import joblib
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import PyPDF2

# -------------------------
# NLTK DOWNLOAD (RUNS ONCE)
# -------------------------
@st.cache_resource(show_spinner=False)
def download_nltk():
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)

    packages = ["punkt_tab", "punkt", "stopwords", "wordnet", "omw-1.4"]
    for pkg in packages:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, download_dir=nltk_data_path)

download_nltk()

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="⚖️",
    layout="centered"
)

# -------------------------
# PREMIUM STYLING
# -------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

body {
    background: linear-gradient(-45deg, #0f172a, #1e3a8a, #0f172a, #111827);
    background-size: 400% 400%;
    animation: gradientMove 18s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.hero-title {
    font-family: 'Cinzel', serif;
    font-size: 60px;
    font-weight: 800;
    letter-spacing: 2px;
    text-align: center;
    background: linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6,#60a5fa);
    background-size: 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 6s linear infinite;
}

@keyframes shine {
    0% {background-position: 0%;}
    100% {background-position: 300%;}
}

.hero-subtitle {
    text-align: center;
    font-size: 20px;
    color: #cbd5e1;
    margin-top: 20px;
    max-width: 750px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}

.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 35px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 15px 40px rgba(0,0,0,0.4);
    margin-top: 40px;
    transition: 0.4s ease;
}

.glass-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}

.result-box {
    text-align: center;
    font-size: 26px;
    font-weight: 600;
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
}

.low {background:#14532d; color:#bbf7d0;}
.medium {background:#78350f; color:#fde68a;}
.high {background:#7f1d1d; color:#fecaca;}

.footer {
    text-align:center;
    margin-top:70px;
    color:#9ca3af;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HERO SECTION
# -------------------------
st.markdown("""
<div style="margin-top:60px;">
    <div class="hero-title">
        Intelligent Contract Risk Intelligence
    </div>
    <div class="hero-subtitle">
        Transform complex legal clauses into clear, data-driven risk insights.
        <br><br>
        Powered by Machine Learning and Natural Language Processing.
        <br><br>
        <strong>Analyze. Understand. Decide with Confidence.</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_models():
    model = joblib.load("risk_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, le = load_models()

# -------------------------
# NLP SETUP
# -------------------------
stop_words = set(stopwords.words('english'))
legal_keep = {"shall", "not", "may", "must"}
stop_words = stop_words - legal_keep
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# -------------------------
# INPUT CARD
# -------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

input_method = st.radio("Choose Input Method:", ["Paste Text", "Upload PDF"], horizontal=True)

user_input = ""

if input_method == "Paste Text":
    user_input = st.text_area(
        "Paste your legal clause below:",
        height=220,
        placeholder="Example: The vendor shall not terminate the agreement without prior written notice..."
    )

elif input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload Contract PDF", type=["pdf"])

    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text

        user_input = extracted_text
        st.success("PDF uploaded successfully.")
        st.text_area("Extracted Text Preview:", user_input[:1000], height=150)

analyze = st.button("🔍 Analyze Risk", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# RESULT SECTION
# -------------------------
if analyze:
    if user_input.strip() == "":
        st.warning("Please enter a contract clause.")
    else:
        with st.spinner("Analyzing legal risk..."):
            cleaned = preprocess_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)
            probabilities = model.predict_proba(vector)

            risk_label = le.inverse_transform(prediction)[0]
            confidence = round(max(probabilities[0]) * 100, 2)

        css_class = risk_label.lower()

        st.markdown(
            f'<div class="result-box {css_class}">'
            f'{risk_label.upper()} RISK<br>'
            f'<span style="font-size:18px;">Confidence: {confidence}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

# -------------------------
# FOOTER
# -------------------------
st.markdown("""
<div class="footer">
    Built with Streamlit • Machine Learning • NLP
</div>
""", unsafe_allow_html=True)