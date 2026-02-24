import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


model = joblib.load("risk_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


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



st.title("📄 Intelligent Contract Risk Analysis")
st.write("Enter a contract clause below to predict its risk level.")

user_input = st.text_area("Contract Clause:", height=150)

if st.button("Analyze Risk"):
    if user_input.strip() == "":
        st.warning("Please enter a clause before analyzing.")
    else:
        cleaned = preprocess_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)
        risk_label = le.inverse_transform(prediction)[0]

        # Color-based output
        if risk_label.lower() == "low":
            st.success(f"Predicted Risk Level: {risk_label.upper()}")
        elif risk_label.lower() == "medium":
            st.warning(f"Predicted Risk Level: {risk_label.upper()}")
        else:
            st.error(f"Predicted Risk Level: {risk_label.upper()}")
st.write("Loaded vectorizer type:", type(vectorizer))
st.write("Has IDF:", hasattr(vectorizer, "idf_"))