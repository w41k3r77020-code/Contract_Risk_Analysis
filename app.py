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
# NLTK DOWNLOAD (RUNS ONLY ONCE)
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
# Page Config
# -------------------------
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="📄",
    layout="centered"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
    <style>
        .main {background-color: #f5f7fa;}
        .title {font-size: 38px;font-weight: 700;text-align: center;color: #1f4e79;}
        .subtitle {text-align: center;color: #555;margin-bottom: 30px;}
        .risk-box {padding: 15px;border-radius: 10px;text-align: center;font-size: 20px;font-weight: bold;margin-top: 20px;}
        .low {background-color: #d4edda;color: #155724;}
        .medium {background-color: #fff3cd;color: #856404;}
        .high {background-color: #f8d7da;color: #721c24;}
        .footer {text-align: center;font-size: 14px;margin-top: 40px;color: grey;}
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model Components (Cached)
# -------------------------
@st.cache_resource
def load_models():
    model = joblib.load("risk_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, le = load_models()

# -------------------------
# NLP Setup
# -------------------------
stop_words = set(stopwords.words('english'))
legal_keep = {"shall", "not", "may", "must"}
stop_words = stop_words - legal_keep
lemmatizer = WordNetLemmatizer()

# -------------------------
# Preprocessing Function
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# -------------------------
# UI Layout
# -------------------------
st.markdown('<div class="title">📄 Intelligent Contract Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered clause-level risk classification system</div>', unsafe_allow_html=True)

# -------------------------
# Input Method Selection
# -------------------------
input_method = st.radio(
    "Choose Input Method:",
    ["Paste Text", "Upload PDF"]
)

user_input = ""

if input_method == "Paste Text":
    user_input = st.text_area("Enter Contract Clause Below:", height=150)

elif input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload Contract PDF", type=["pdf"])

    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        extracted_text = ""

        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text

        user_input = extracted_text
        st.success("PDF uploaded and text extracted successfully!")
        st.text_area("Extracted Text Preview:", user_input[:1000], height=150)

if st.button("Analyze Risk"):

    if user_input.strip() == "":
        st.warning("Please enter a contract clause to analyze.")
    else:
        cleaned = preprocess_text(user_input)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)
        probabilities = model.predict_proba(vector)

        risk_label = le.inverse_transform(prediction)[0]
        confidence = round(max(probabilities[0]) * 100, 2)

        # Display Result with Color
        if risk_label.lower() == "low":
            st.markdown(f'<div class="risk-box low">LOW RISK<br>Confidence: {confidence}%</div>', unsafe_allow_html=True)
        elif risk_label.lower() == "medium":
            st.markdown(f'<div class="risk-box medium">MEDIUM RISK<br>Confidence: {confidence}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-box high">HIGH RISK<br>Confidence: {confidence}%</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Built with ❤️ using Streamlit & Scikit-Learn</div>', unsafe_allow_html=True)