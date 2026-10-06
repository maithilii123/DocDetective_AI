import re
import joblib

from pathlib import Path
from pypdf import PdfReader

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"

CATEGORIES = [
    "Resume",
    "Invoice",
    "Certificate",
    "ID_Document",
    "Report",
    "Other"
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Clean extracted text."""

    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):
    """Extract text from a PDF."""

    text = ""

    try:
        reader = PdfReader(str(pdf_path))

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    except Exception as e:

        print(f"Error reading {pdf_path}: {e}")

    return text


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    documents = []
    labels = []

    print("\n========================================")
    print("LOADING DATASET")
    print("========================================\n")

    for category in CATEGORIES:

        category_path = Path(DATASET_DIR) / category

        if not category_path.exists():

            print(f"Folder not found: {category_path}")
            continue

        pdf_files = list(category_path.glob("*.pdf"))

        print(
            f"{category}: {len(pdf_files)} PDF files"
        )

        for pdf_file in pdf_files:

            text = extract_pdf_text(pdf_file)

            if text.strip():

                cleaned = clean_text(text)

                documents.append(cleaned)
                labels.append(category)

            else:

                print(
                    f"Warning: No text found in {pdf_file}"
                )

    return documents, labels


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    # Load documents
    documents, labels = load_dataset()

    print("\nTotal documents:", len(documents))

    # Check dataset
    if len(documents) < 6:

        print(
            "\nERROR: Not enough documents."
        )

        print(
            "Please add more PDF files to your dataset folders."
        )

        return

    # Check categories
    unique_categories = set(labels)

    print(
        "Categories found:",
        len(unique_categories)
    )

    print(
        "Category names:",
        sorted(unique_categories)
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print("\n========================================")
    print("TRAIN / TEST SPLIT")
    print("========================================")

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            documents,
            labels,
            test_size=0.33,
            random_state=42,
            stratify=labels
        )

    except ValueError as e:

        print("\nDataset split error:")
        print(e)

        print(
            "\nYou need at least 2 documents "
            "in every category."
        )

        return

    print(
        f"Training documents: {len(X_train)}"
    )

    print(
        f"Testing documents: {len(X_test)}"
    )

    # ========================================================
    # TF-IDF
    # ========================================================

    print("\n========================================")
    print("CREATING TF-IDF FEATURES")
    print("========================================")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print(
        "TF-IDF features:",
        X_train_tfidf.shape[1]
    )

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    print("\n========================================")
    print("TRAINING MODEL")
    print("========================================")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    print(
        "Logistic Regression training completed."
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = model.predict(
        X_test_tfidf
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========================================")
    print("MODEL EVALUATION")
    print("========================================")

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        "document_classifier.pkl"
    )

    joblib.dump(
        vectorizer,
        "tfidf_vectorizer.pkl"
    )

    print("\n========================================")
    print("MODEL SAVED")
    print("========================================")

    print(
        "\nCreated:"
    )

    print(
        "document_classifier.pkl"
    )

    print(
        "tfidf_vectorizer.pkl"
    )

    print(
        "\nTraining completed successfully! ✅"
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()