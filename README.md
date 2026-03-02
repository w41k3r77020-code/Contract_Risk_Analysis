# Intelligent Contract Risk Classification  
Milestone 1 – Classical NLP & Machine Learning System

---

## 1. Project Overview

This project implements a machine learning-based contract clause risk classification system designed to automatically identify potentially risky clauses in legal documents.

The system applies classical Natural Language Processing (NLP) techniques and supervised learning models to perform clause-level risk prediction without relying on Large Language Models (LLMs).

This work corresponds to Milestone 1 of Project 7 – AI-Driven Legal Document Analysis.

---

## 2. Objective

The objective of this project is to design and implement a clause-level risk classification system that:

- Accepts contract text or PDF input  
- Performs structured preprocessing  
- Extracts meaningful textual features  
- Classifies clauses into predefined risk categories  
- Produces structured and interpretable risk predictions  

---

## 3. Technology Stack

- Python  
- Scikit-learn  
- NLTK  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  
- Joblib (Model Serialization)  

---

## 4. Dataset Description

The dataset consists of labeled legal contract clauses categorized into the following risk levels:

- High Risk  
- Medium Risk  
- Low Risk  

The problem is formulated as a multi-class text classification task at the clause level.

---

## 5. Data Preprocessing Pipeline

The preprocessing workflow was designed to preserve legal semantics while reducing textual noise.

### Steps Implemented

1. Lowercasing normalization  
2. Removal of punctuation  
3. Tokenization using NLTK  
4. Stopword removal  
5. Preservation of legally significant keywords such as:
   - shall  
   - must  
   - may  
   - not  
6. Lemmatization using WordNet  
7. Text reconstruction prior to vectorization  

Preserving legal obligation terms ensured improved semantic representation of contractual clauses.

---

## 6. Feature Engineering

TF-IDF (Term Frequency–Inverse Document Frequency) vectorization was applied with the following configuration:

- Maximum Features: 5000  
- N-gram Range: (1,2)  
- Minimum Document Frequency: 2  

The inclusion of bigrams allowed the model to capture contextual legal expressions such as:

- breach contract  
- terminate agreement  
- indemnify party  

The resulting feature matrix was high-dimensional and sparse, making it well-suited for linear classification models.

---

## 7. Models Implemented

### Logistic Regression
- max_iter = 1000  

### Decision Tree Classifier
- max_depth = 20  
- random_state = 42  

---

## 8. Training Strategy

- 80–20 Train-Test Split  
- Stratified Sampling to preserve class distribution  
- Random State fixed at 42 for reproducibility  

---

## 9. Evaluation Metrics

The system was evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 10. Results

| Model               | Accuracy | F1 Score |
|---------------------|----------|----------|
| Logistic Regression | 0.86     | 0.86     |
| Decision Tree       | 0.80     | 0.80     |

Logistic Regression demonstrated superior generalization performance due to its robustness in high-dimensional sparse TF-IDF feature spaces. The Decision Tree model exhibited slight overfitting tendencies.

---

## 11. Model Serialization

The following components were serialized using Joblib:

- Trained Logistic Regression model  
- TF-IDF vectorizer  
- Label encoder  

These artifacts enable seamless integration into a deployment-ready user interface for real-time clause-level risk prediction.

---

## 12. User Interface Integration

The system is structured for integration with a user interface capable of:

- Accepting contract text or PDF input  
- Performing automated preprocessing  
- Generating clause-level risk predictions  
- Highlighting high-risk clauses  

---

## 13. Project Structure
├── CONTRACT_RISK_CLASSIFICATION.ipynb
├── model.pkl
├── vectorizer.pkl
├── label_encoder.pkl
├── README.md


## 14. Disclaimer

This system is developed for academic and research purposes only and does not constitute legal advice.

## 15. Deployed URL
https://contractriskanalysis-l3qvfbvsojfij8hek3khzk.streamlit.app/

## Application Preview

![Dashboard Screenshot](images/confusion_matrix.png)

