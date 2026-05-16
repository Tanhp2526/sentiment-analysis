"""
=============================================================
BUOC 4: PREDICT REVIEW MOI
=============================================================
File nay lam 2 viec:
1. Load model + vectorizer da luu (tu step3)
2. Predict cam xuc cho review moi (nhap tu ban phim)

Day la buoc "su dung" model da train:
-> Khong can train lai
-> Chi can load file .pkl va predict
=============================================================
"""

import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()


def clean_text(text):
    """Lam sach text - GIONG HET ham trong step2"""
    text = re.sub('<.*?>', ' ', text)           # Xoa HTML
    text = text.lower()                          # Lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)       # Xoa ky tu dac biet
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in STOP_WORDS]
    return ' '.join(words)


# ===== LOAD MODEL + VECTORIZER =====
print("=" * 60)
print("BUOC 4: Predict review moi")
print("=" * 60)

# Load model va vectorizer da luu tu step3
# joblib.load() = doc file .pkl va tra ve object Python
model = joblib.load(os.path.join(ROOT_DIR, "models", "best_model.pkl"))
tfidf = joblib.load(os.path.join(ROOT_DIR, "models", "tfidf_vectorizer.pkl"))
print(f"\nDa load model va vectorizer!")
print(f"   Model: {type(model).__name__}")


# ===== THU PREDICT MOT SO CAU MAU =====
print(f"\n{'=' * 60}")
print("Thu predict voi cac review mau:")
print("=" * 60)

sample_reviews = [
    "This movie was absolutely wonderful! Great acting and beautiful story.",
    "Terrible film. Waste of time. The worst movie I have ever seen.",
    "It was okay, nothing special but not bad either.",
    "The cinematography was stunning but the plot was boring and predictable.",
    "I love this movie so much! Definitely my favorite film of the year!",
    "Awful acting, terrible script, do not waste your money on this garbage.",
]

for review in sample_reviews:
    # Buoc 1: Lam sach text (GIONG step2)
    clean = clean_text(review)

    # Buoc 2: Chuyen thanh TF-IDF vector
    # QUAN TRONG: dung .transform() (KHONG phai .fit_transform())
    # Vi vocabulary da duoc xay tu train data roi
    vector = tfidf.transform([clean])

    # Buoc 3: Predict
    prediction = model.predict(vector)[0]
    label = "POSITIVE" if prediction == 1 else "NEGATIVE"

    # In ket qua
    print(f"\n   Review: \"{review[:70]}...\"" if len(review) > 70 else f"\n   Review: \"{review}\"")
    print(f"   Clean:  \"{clean[:70]}...\"" if len(clean) > 70 else f"   Clean:  \"{clean}\"")
    print(f"   => {label} {'[+]' if prediction == 1 else '[-]'}")


# ===== CHE DO NHAP TAY =====
print(f"\n{'=' * 60}")
print("Nhap review de predict (go 'quit' de thoat):")
print("=" * 60)

while True:
    print()
    user_input = input(">> Nhap review: ").strip()

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Bye!")
        break

    if not user_input:
        print("   (Review rong, thu lai)")
        continue

    clean = clean_text(user_input)
    vector = tfidf.transform([clean])
    prediction = model.predict(vector)[0]
    label = "POSITIVE" if prediction == 1 else "NEGATIVE"

    print(f"   Clean: \"{clean}\"")
    print(f"   => {label} {'[+]' if prediction == 1 else '[-]'}")
