"""
=============================================================
BUOC 5: STREAMLIT WEB DEMO
=============================================================
Tao web app de:
- Nguoi dung nhap 1 review bat ky
- Model predict positive/negative
- Hien thi ket qua dep, truc quan

Chay bang lenh: streamlit run app.py
=============================================================
"""

import streamlit as st
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()


def clean_text(text):
    """Lam sach text - GIONG HET ham trong step2"""
    text = re.sub('<.*?>', ' ', text)
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in STOP_WORDS]
    return ' '.join(words)


# ===== LOAD MODEL (chi load 1 lan, cache lai) =====
# st.cache_resource: Streamlit luu model vao cache
# -> Lan dau load tu file, lan sau dung lai tu cache (nhanh hon)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(ROOT_DIR, "models", "best_model.pkl"))
    tfidf = joblib.load(os.path.join(ROOT_DIR, "models", "tfidf_vectorizer.pkl"))
    return model, tfidf

model, tfidf = load_model()


# ===== GIAO DIEN WEB =====
# st.set_page_config: cau hinh trang web
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Tieu de
st.title(" IMDB Sentiment Analysis")
st.markdown("Nhap mot review phim bat ky, model se du doan **Positive** hay **Negative**.")
st.markdown("---")

# O nhap text
# st.text_area: tao o nhap text lon (nhieu dong)
user_review = st.text_area(
    "Nhap review cua ban:",
    placeholder="Type your movie review here...",
    height=150
)

# Nut predict
if st.button("Predict Sentiment", type="primary"):
    if user_review.strip():
        # Xu ly
        clean = clean_text(user_review)
        vector = tfidf.transform([clean])
        prediction = model.predict(vector)[0]

        # Hien thi ket qua
        st.markdown("---")
        if prediction == 1:
            st.success("##  POSITIVE")
            st.balloons()  # Hieu ung bong bay
        else:
            st.error("##  NEGATIVE")

        # Hien thi chi tiet
        with st.expander("Xem chi tiet xu ly"):
            st.write(f"**Original:** {user_review[:200]}...")
            st.write(f"**After cleaning:** {clean[:200]}...")
            st.write(f"**Model:** {type(model).__name__}")
    else:
        st.warning("Vui long nhap review truoc khi predict!")


# ===== VI DU MAU =====
st.markdown("---")
st.markdown("### Thu voi cac review mau:")

# Tao 2 cot
col1, col2 = st.columns(2)

with col1:
    if st.button(" This movie is amazing!"):
        clean = clean_text("This movie is amazing and wonderful! Great acting!")
        vector = tfidf.transform([clean])
        pred = model.predict(vector)[0]
        if pred == 1:
            st.success("POSITIVE ")
        else:
            st.error("NEGATIVE ")

    if st.button(" Brilliant masterpiece"):
        clean = clean_text("A brilliant masterpiece with stunning visuals and incredible performances")
        vector = tfidf.transform([clean])
        pred = model.predict(vector)[0]
        if pred == 1:
            st.success("POSITIVE ")
        else:
            st.error("NEGATIVE ")

with col2:
    if st.button(" Terrible waste of time"):
        clean = clean_text("Terrible movie. Complete waste of time. Awful acting.")
        vector = tfidf.transform([clean])
        pred = model.predict(vector)[0]
        if pred == 1:
            st.success("POSITIVE ")
        else:
            st.error("NEGATIVE ")

    if st.button(" Boring and predictable"):
        clean = clean_text("The most boring and predictable movie I have ever watched. Do not recommend.")
        vector = tfidf.transform([clean])
        pred = model.predict(vector)[0]
        if pred == 1:
            st.success("POSITIVE ")
        else:
            st.error("NEGATIVE ")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Scikit-learn + Streamlit | IMDB Dataset (50K reviews)"
    "</div>",
    unsafe_allow_html=True
)
