import re
import joblib
import pytesseract

import streamlit as st

from pypdf import PdfReader
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocDetective AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown("""
<style>

    /* Main page */
    .stApp {
        background-color: #f7f9fc;
    }

    .main .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #172554, #2563eb);
        padding: 35px 40px;
        border-radius: 22px;
        margin-bottom: 28px;
        color: white;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.18);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
        font-weight: 750;
    }

    .hero p {
        font-size: 17px;
        margin: 0;
        opacity: 0.92;
    }

    /* Section titles */
    .section-title {
        font-size: 23px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 12px;
        color: #172033;
    }

    /* Upload card */
    .upload-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    /* Result card */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        border: 1px solid #dbe3ef;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.07);
        margin-top: 15px;
        margin-bottom: 25px;
    }

    .result-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .result-type {
        font-size: 32px;
        font-weight: 750;
        color: #1d4ed8;
        margin-top: 5px;
        margin-bottom: 20px;
    }

    /* Confidence */
    .confidence-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
    }

    .confidence-number {
        font-size: 34px;
        font-weight: 750;
        color: #1d4ed8;
    }

    .confidence-label {
        color: #64748b;
        font-size: 14px;
    }

    /* File information */
    .file-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px 18px;
        border-radius: 12px;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* Probability cards */
    .prob-title {
        font-size: 16px;
        font-weight: 650;
        color: #334155;
        margin-bottom: 4px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 750;
        color: #172033;
    }

    .sidebar-card {
        background: #f8fafc;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 650;
        border: none;
        background: #2563eb;
        color: white;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
    }

    /* Divider */
    hr {
        margin-top: 30px;
        margin-bottom: 30px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🔎 DocDetective AI</h1>
    <p>AI-Powered Document Classification & Analysis System</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="upload-card">
    <div class="section-title">📤 Upload Your Document</div>
    <p>
        Upload a PDF or image and let DocDetective AI
        identify the document category using machine learning.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = "document_classifier.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    return model, vectorizer


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    text = ""

    try:

        reader = PdfReader(
            uploaded_file
        )

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + " "

    except Exception as e:

        st.error(
            f"PDF reading error: {e}"
        )

    return text


# ============================================================
# IMAGE OCR
# ============================================================

def extract_image_text(uploaded_file):

    try:

        image = Image.open(
            uploaded_file
        )

        text = pytesseract.image_to_string(
            image
        )

        return text

    except Exception as e:

        st.error(
            f"OCR error: {e}"
        )

        return ""


# ============================================================
# DOCUMENT TEXT EXTRACTION
# ============================================================

def extract_text(uploaded_file):

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf_text(
            uploaded_file
        )

    elif (
        filename.endswith(".png")
        or filename.endswith(".jpg")
        or filename.endswith(".jpeg")
    ):

        return extract_image_text(
            uploaded_file
        )

    return ""


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a PDF or image",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# ANALYZE DOCUMENT
# ============================================================

if uploaded_file is not None:

    st.markdown(
        f"""
        <div class="file-box">
            📄 <b>Selected File:</b> {uploaded_file.name}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "🔍 Analyze Document"
    ):

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        with st.spinner(
            "Extracting document text..."
        ):

            text = extract_text(
                uploaded_file
            )

        if not text.strip():

            st.error(
                "No text could be extracted."
            )

            st.info(
                "Try a text-based PDF or a clearer image."
            )

            st.stop()

        # ----------------------------------------------------
        # Clean text
        # ----------------------------------------------------

        cleaned_text = clean_text(
            text
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        try:

            model, vectorizer = load_model()

        except FileNotFoundError:

            st.error(
                "Trained model files were not found."
            )

            st.warning(
                "Run 'python train_model.py' first."
            )

            st.stop()

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        text_vector = vectorizer.transform(
            [cleaned_text]
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            text_vector
        )[0]

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            text_vector
        )[0]

        confidence = (
            max(probabilities) * 100
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("📄 Prediction Result")

        st.success(
            f"Predicted Document Type: {prediction}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            float(confidence / 100)
        )

        if confidence >= 70:
            st.success(
                "🟢 High confidence prediction"
        )

        elif confidence >= 40:
            st.warning(
                "🟡 Medium confidence prediction"
        )

        else:
            st.error(
                "🔴 Low confidence prediction"
        )
        # ----------------------------------------------------
        # CLASS PROBABILITIES
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Class Probabilities</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Probability distribution across all supported document categories."
        )

        classes = model.classes_

        for class_name, probability in zip(
            classes,
            probabilities
        ):

            percentage = (
                probability * 100
            )

            col_a, col_b = st.columns([4, 1])

            with col_a:

                st.markdown(
                    f'<div class="prob-title">{class_name}</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    float(probability)
                )

            with col_b:

                st.markdown(
                    f"""
                    <div style="
                        text-align:right;
                        font-weight:700;
                        padding-top:4px;
                        color:#334155;
                    ">
                        {percentage:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------
        # EXTRACTED TEXT
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">📃 Extracted Text</div>',
            unsafe_allow_html=True
        )

        with st.expander(
            "Click to view extracted document text"
        ):

            st.text(
                text[:5000]
            )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-title">🔎 DocDetective AI</div>',
    unsafe_allow_html=True
)

st.sidebar.write(
    "AI-powered document classification system."
)

st.sidebar.divider()


st.sidebar.markdown(
    '<div class="sidebar-card"><b>📁 Supported Documents</b><br><br>'
    '📄 Resume<br>'
    '🧾 Invoice<br>'
    '🏆 Certificate<br>'
    '🪪 ID Document<br>'
    '📑 Report<br>'
    '📂 Other</div>',
    unsafe_allow_html=True
)


st.sidebar.markdown(
    '<div class="sidebar-card"><b>🤖 Machine Learning</b><br><br>'
    'TF-IDF Vectorization<br>'
    'Logistic Regression<br>'
    'OCR with Tesseract</div>',
    unsafe_allow_html=True
)


st.sidebar.info(
    "💡 Upload a clear document for better classification results."
)


st.sidebar.divider()

st.sidebar.caption(
    "DocDetective AI • Document Intelligence"
)