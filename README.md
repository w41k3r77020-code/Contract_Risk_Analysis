# ⚖️ Intelligent Contract Risk Analysis using RAG & Agentic AI

## 🚀 From ML Classification to Agentic Legal Reasoning

---

## 📌 Project Overview

This project presents an **AI-powered system for analyzing legal contracts** and identifying potential risks at the clause level.

It evolves from a traditional Machine Learning approach to a modern **Agentic AI system** that combines:

* Retrieval-Augmented Generation (RAG)
* Large Language Models (LLMs)
* LangGraph-based agent workflows

The system not only predicts risk but also provides **explanations and contextual insights**.

---

## 🧠 Key Features

* 📄 Upload contract (PDF) or paste text
* 🔍 Clause-level risk analysis
* ⚠️ Risk classification (High / Medium / Low)
* 📚 Context retrieval using FAISS (RAG)
* 🧠 LLM-based reasoning (Groq API)
* 🔁 Agentic workflow using LangGraph
* 📊 Structured risk report
* 🌐 Deployed using Streamlit

---

## 🚀 How It Works

```text
Input (PDF/Text)
        ↓
Clause Segmentation (utils/parser.py)
        ↓
RAG Retrieval (rag/retriever.py)
        ↓
LLM Reasoning (rag/llm_rag.py)
        ↓
Agent Workflow (agent/graph.py)
        ↓
Final Risk Report
```

---

## 📊 Milestone 1: ML-Based Risk Classification

### 🔹 Approach

* Text preprocessing (cleaning, tokenization)
* Feature extraction using TF-IDF
* Model: Logistic Regression

### 🔹 Files

* `risk_model.pkl`
* `label_encoder.pkl`
* `tfidf_vectorizer.pkl`

### 🔹 Output

* Risk classification (High / Medium / Low)

### 🔹 Limitation

* No explanation or reasoning
* No contextual understanding

## 📊 Model Evaluation

### 🔹 Performance Summary

- **Accuracy:** 86.7%  
- **Macro F1 Score:** ~0.86  
- **Class Distribution:** Balanced  

### 🔹 Class-wise Performance Insights

- ✅ Strong performance on Low-risk (Class 0)  
  - 745 clauses correctly classified  

- ✅ Strong performance on High-risk (Class 2)  
  - 605 clauses correctly classified  

- ⚠️ Moderate confusion in Medium-risk (Class 1)  

### 🔹 Key Observations

- Strong performance on extreme classes  
- Medium-risk shows overlap  
- Reliable baseline model  


---

## 🚀 Milestone 2: Agentic AI + RAG System

### 🔹 1. Clause Segmentation

* Implemented in `utils/parser.py`
* Splits contracts into structured clauses

---

### 🔹 2. RAG Pipeline

* Embeddings: sentence-transformers
* Vector store: FAISS
* Files:

  * `faiss.index`
  * `metadata.pkl`
  * `build_index.py`

```text
Query → Embedding → Similar Clauses → Context
```

---

### 🔹 3. LLM Reasoning

* Implemented in `rag/llm_rag.py`
* Uses Groq API
* Generates:

  * Risk level
  * Explanation
  * Recommendation

---

### 🔹 4. Agentic Workflow (LangGraph)

* Implemented in `agent/graph.py`

```text
Extract → Analyze → Loop → Report
```

* Processes clauses iteratively
* Maintains state
* Generates final report

---

## 🖥️ User Interface

Built using **Streamlit (`app.py`)**

### Features:

* Upload PDF or paste text
* View contract content
* Risk dashboard
* Clause-level results
* Export report

---

## 🌐 Deployment

Deployed on Streamlit Cloud

🔗 **Live App:** [https://your-app-link.streamlit.app](https://contractriskanalysis-l3qvfbvsojfij8hek3khzk.streamlit.app/)

---

## ⚙️ Tech Stack

### 🔹 NLP & ML

* scikit-learn
* nltk
* TF-IDF

### 🔹 RAG

* sentence-transformers
* FAISS

### 🔹 Agentic AI

* LangGraph
* LangChain

### 🔹 LLM

* Groq API

### 🔹 Backend API & Frontend Architecture

* **Backend:** FastAPI, Uvicorn, Python-Multipart
* **Frontend:** React 19, Vite, Tailwind CSS v4, Framer Motion, Recharts, Lucide Icons

### 🔹 Others

* PyPDF
* pandas, numpy

---

## 📁 Project Structure

```text
Contract_Risk_Analysis/
│
├── main.py                # FastAPI REST API layer
├── app.py                 # Original Streamlit app (preserved)
├── requirements.txt       # Backend dependencies
├── Procfile               # Deployment config for Render/Railway
├── .env.example           # Environment template
├── README.md
│
├── frontend/              # Deploy-ready React + Tailwind frontend (Vercel)
│   ├── src/
│   │   ├── components/    # Navbar, LandingHero, IntakeSection, ResultsView, Tabs
│   │   ├── services/      # api.js client
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── vercel.json        # Vercel SPA rewrites
│   ├── package.json
│   └── vite.config.js
│
├── agent/
│   └── graph.py           # LangGraph workflow (unchanged)
│
├── rag/
│   ├── retriever.py       # FAISS retriever (unchanged)
│   ├── llm_rag.py         # Groq LLM reasoning (unchanged)
│   ├── build_index.py
│   ├── faiss.index
│   └── metadata.pkl
│
├── utils/
│   └── parser.py          # PDF extraction & clause segmentation (unchanged)
│
├── risk_model.pkl
├── label_encoder.pkl
└── tfidf_vectorizer.pkl
```

---

## 🔐 Environment Setup

Create a `.env` file in the root directory:

```text
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Run Locally

### 1. Run FastAPI Backend

```bash
# In project root
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend API will be accessible at: `http://localhost:8000` (docs at `http://localhost:8000/docs`).

### 2. Run React Frontend

```bash
# In frontend directory
cd frontend
npm install
npm run dev
```

Frontend will be accessible at: `http://localhost:5173`.

---

## 🚀 Deployment Guide

### Deploying Frontend on Vercel
1. Push repository to GitHub.
2. In Vercel, import the repository and set **Root Directory** to `frontend`.
3. Framework Preset will auto-detect as **Vite**.
4. Add Environment Variable:
   - `VITE_API_URL`: URL of your deployed FastAPI backend (e.g., `https://your-app.onrender.com`).
5. Deploy!

### Deploying Backend on Render / Railway
1. Create a new Web Service pointing to this repository.
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variable:
   - `GROQ_API_KEY`: Your Groq API key.
5. Deploy!


---

## 📊 Evaluation Alignment

✔ RAG implementation
✔ Agentic AI (LangGraph)
✔ LLM-based reasoning
✔ Clean architecture
✔ Deployed UI
✔ Explainable outputs

---

## ⚠️ Disclaimer

This system provides AI-generated insights and **does not constitute legal advice**.

---

## 👥 Team Members

* Bulbul Agarwalla
* Ganga Raghuwanshi
* Anuradha Raghuwanshi
* Alisha Gupta

---

## 🎯 Future Work

* Clause type classification
* Highlight clauses in PDF
* Multi-contract comparison
* Fine-tuned legal LLM

---

## 🧠 Key Insight

> This project demonstrates how RAG combined with agentic workflows enables explainable, context-aware contract analysis beyond traditional ML approaches.


