# 📄 DocDetective AI

DocDetective AI is an AI-based document classification and analysis system that automatically identifies the type of an uploaded document.

## 🚀 Features

- Upload a document for classification
- Automatically predicts the document type
- Displays prediction confidence
- Shows class-wise probability scores
- Supports multiple document categories
- Simple and user-friendly Streamlit interface

## 📂 Supported Document Types

The system can classify documents into the following categories:

- Certificate
- ID Document
- Invoice
- Other
- Report
- Resume

## 🛠️ Technologies Used

- Python
- Streamlit
- Machine Learning
- TF-IDF Vectorization
- Scikit-learn
- PDF Processing
- OCR / Tesseract

## ⚙️ Project Structure

```text
DocDetective_AI/
│
├── app.py
├── train_model.py
├── requirements.txt
├── document_classifier.pkl
├── tfidf_vectorizer.pkl
│
├── dataset/
│   ├── Certificate/
│   ├── ID_Document/
│   ├── Invoice/
│   ├── Other/
│   ├── Report/
│   └── Resume/
│
└── README.md
