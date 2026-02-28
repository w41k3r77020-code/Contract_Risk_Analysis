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
st.set_page_config(page_title="Contract Risk Analyzer", page_icon="📄", layout="wide")

# -------------------------
# Modern CSS
# -------------------------
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {
    background: linear-gradient(135deg,#1f4e79,#4f8cc9);
    padding: 35px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}
.risk {
    font-size: 28px;
    font-weight: 700;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
}
.low {background:#d4edda;color:#155724;}
.medium {background:#fff3cd;color:#856404;}
.high {background:#f8d7da;color:#721c24;}
.footer {text-align:center;color:gray;margin-top:40px}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Models
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
# Preprocess
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# -------------------------
# HERO
# -------------------------
st.markdown('<div class="hero"><h1>📄 Intelligent Contract Risk Analyzer</h1><p>AI-powered clause level legal risk detection</p></div>', unsafe_allow_html=True)

# -------------------------
# INPUT SECTION
# -------------------------
container = st.container()

with container:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    input_method = st.radio("Choose Input Method:", ["Paste Text", "Upload PDF"])

    user_input = ""

    if input_method == "Paste Text":
        user_input = st.text_area(
            "Paste legal clause here",
            height=220,
            placeholder="Example: The vendor shall not terminate the agreement without prior written notice..."
        )

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

    analyze = st.button("🔍 Analyze Risk", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------
    # RESULT SECTION
    # -------------------------
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Risk Assessment")

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
                f'<div class="risk {css_class}">{risk_label.upper()} RISK<br><span style="font-size:18px">Confidence: {confidence}%</span></div>',
                unsafe_allow_html=True
            )

            st.progress(int(confidence))
    else:
        st.info("Result will appear here after analysis")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Built using Streamlit • Machine Learning • NLP</div>', unsafe_allow_html=True)