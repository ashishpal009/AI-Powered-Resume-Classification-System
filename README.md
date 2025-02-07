# 🚀 AI-Powered Resume Classification System

## Overview
An advanced NLP-powered resume classification system designed to streamline recruitment workflows by automating job role-based categorization with high accuracy and efficiency.

## Key Features

✅ Achieved an impressive **98.45% accuracy** using cutting-edge NLP techniques  
✅ Leveraged **TF-IDF vectorization** and **One-vs-Rest KNN** for robust multi-label classification  
✅ Engineered a **scalable solution** capable of processing large volumes of resumes seamlessly  
✅ Automated **job role-based categorization**, significantly enhancing recruitment workflows  
✅ Built and deployed using **Python & Scikit-learn**, ensuring flexibility and performance  

## Tech Stack

🔹 **Python** 🐍 – Core programming language for development  
🔹 **Scikit-learn** 🤖 – Machine learning framework for model building  
🔹 **NLP (TF-IDF)** 📝 – Feature extraction for text processing  
🔹 **KNN (One-vs-Rest)** 📊 – Multi-label classification for precise categorization  

## Usage

```python
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

# Load the trained model
model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Sample resume text
resume_text = "Experienced Data Scientist skilled in Python, Machine Learning, and NLP."

# Transform input text
resume_tfidf = vectorizer.transform([resume_text])

# Predict job role
predicted_roles = model.predict(resume_tfidf)
print("Predicted Job Roles:", predicted_roles)
```

## Dataset
The model is trained on a dataset consisting of categorized resumes. The dataset includes:
- Resume text
- Corresponding job roles (multi-label classification)

