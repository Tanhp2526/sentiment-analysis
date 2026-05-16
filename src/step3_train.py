"""
=============================================================
BUOC 3: CHUYEN TEXT -> SO (TF-IDF) + TRAIN MODELS
=============================================================
File nay lam 4 viec:
1. Doc data da xu ly (train_clean.csv, test_clean.csv)
2. Chuyen text thanh so bang TF-IDF
3. Train 4 models: Naive Bayes, Logistic Regression, Random Forest, SVM
4. So sanh ket qua, chon model tot nhat, luu model

Day la buoc QUAN TRONG NHAT:
-> Text "movi absolut amaz best film ever" -> vector so [0.3, 0, 0.5, ...]
-> Vector so nay moi la input cho model ML
=============================================================
"""

# ===== IMPORT THU VIEN =====
import pandas as pd
import numpy as np
import os
import time     # Do thoi gian train
import joblib   # Luu/load model

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# TF-IDF: chuyen text thanh so (da hoc o file giai thich)
from sklearn.feature_extraction.text import TfidfVectorizer

# 4 models se train
from sklearn.naive_bayes import MultinomialNB          # Naive Bayes
from sklearn.linear_model import LogisticRegression    # Logistic Regression
from sklearn.ensemble import RandomForestClassifier    # Random Forest
from sklearn.svm import LinearSVC                      # SVM (dung LinearSVC cho nhanh)

# Metrics: danh gia model
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Ve bieu do
import matplotlib.pyplot as plt
import seaborn as sns


# ===== BUOC 3.1: DOC DATA DA XU LY =====
print("=" * 60)
print("BUOC 3: TF-IDF + Train Models")
print("=" * 60)

train_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "train_clean.csv"))
test_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "test_clean.csv"))

print(f"\nDoc xong:")
print(f"   Train: {train_df.shape[0]:,} reviews")
print(f"   Test:  {test_df.shape[0]:,} reviews")

# Tach text va label ra rieng
# X = input (text), y = output (label)
# Quy uoc trong ML: X = features, y = target
X_train_text = train_df["clean_review"].astype(str)    # Dam bao la string
X_test_text = test_df["clean_review"].astype(str)
y_train = train_df["label"]
y_test = test_df["label"]


# ===== BUOC 3.2: CHUYEN TEXT -> SO BANG TF-IDF =====
print(f"\n{'=' * 60}")
print("BUOC 3.2: Chuyen text -> so bang TF-IDF")
print("=" * 60)

# Tao TfidfVectorizer voi cac thong so:
# - max_features=10000: chi giu 10,000 tu quan trong nhat
#   Tai sao? Vi dataset co the co 50,000+ tu khac nhau
#   Giu het -> ma tran qua lon, cham, va nhieu tu la rac
#   10,000 tu pho bien nhat la du de model hoc tot
#
# - ngram_range=(1,2): xet ca tu don (1-gram) va cap tu (2-gram)
#   1-gram: "good", "bad", "movie"
#   2-gram: "not good", "very bad", "great movie"
#   Tai sao? Vi "not good" co nghia NGUOC voi "good"
#   Neu chi xet 1-gram, model khong phan biet duoc!
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# fit_transform tren TRAIN data:
# - fit: xay vocabulary tu train data (10,000 tu/cum tu quan trong nhat)
# - transform: tinh TF-IDF cho moi review -> ma tran so
print(f"\nDang chuyen {X_train_text.shape[0]:,} train reviews thanh TF-IDF vectors...")
X_train = tfidf.fit_transform(X_train_text)
print(f">> Xong! Ma tran: {X_train.shape[0]:,} reviews x {X_train.shape[1]:,} features")

# transform tren TEST data (KHONG fit lai!)
# Dung vocabulary da xay tu train data
print(f"Dang chuyen {X_test_text.shape[0]:,} test reviews...")
X_test = tfidf.transform(X_test_text)
print(f">> Xong! Ma tran: {X_test.shape[0]:,} reviews x {X_test.shape[1]:,} features")

# Xem 10 tu/cum tu co TF-IDF cao nhat trong vocabulary
feature_names = tfidf.get_feature_names_out()
print(f"\nMot vai tu trong vocabulary:")
print(f"   {feature_names[:10].tolist()}")
print(f"   ... ({len(feature_names):,} tu tong cong)")


# ===== BUOC 3.3: TRAIN 4 MODELS =====
print(f"\n{'=' * 60}")
print("BUOC 3.3: Train 4 Models")
print("=" * 60)

# Tao dict chua 4 models
# Tai sao train nhieu model?
# -> Khong ai biet truoc model nao tot nhat cho dataset cu the
# -> Thu nhieu -> so sanh -> chon tot nhat = cach tiep can khoa hoc
models = {
    "Naive Bayes": MultinomialNB(),
    # MultinomialNB: model xac suat, rat nhanh, thuong la baseline tot cho text

    "Logistic Regression": LogisticRegression(max_iter=1000),
    # LogisticRegression: model tuyen tinh, nhanh, interpretable
    # max_iter=1000: so vong lap toi da de hoi tu (mac dinh 100 co the khong du)

    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    # RandomForest: ensemble (tap hop nhieu decision tree)
    # n_estimators=100: dung 100 cay quyet dinh
    # n_jobs=-1: dung tat ca CPU cores de train nhanh hon

    "SVM (Linear)": LinearSVC(max_iter=2000),
    # LinearSVC: Support Vector Machine voi kernel tuyen tinh
    # Rat tot voi data cao chieu (text co nhieu features)
    # max_iter=2000: tang so vong lap de dam bao hoi tu
}

# Dict luu ket qua
results = {}

# Train tung model
for name, model in models.items():
    print(f"\n--- Training: {name} ---")

    # Do thoi gian train
    start_time = time.time()

    # Train model
    # .fit(X, y) = cho model hoc tu data
    # X_train = ma tran TF-IDF (so), y_train = nhan (0 hoac 1)
    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    # Predict tren test data
    # .predict(X) = du doan nhan cho data moi
    y_pred = model.predict(X_test)

    # Tinh metrics
    acc = accuracy_score(y_test, y_pred)    # % dung tong the
    f1 = f1_score(y_test, y_pred)           # F1-score (metric chinh)

    # Luu ket qua
    results[name] = {
        "model": model,
        "accuracy": acc,
        "f1_score": f1,
        "train_time": train_time,
        "y_pred": y_pred
    }

    print(f"   Accuracy:   {acc:.4f} ({acc*100:.2f}%)")
    print(f"   F1-Score:   {f1:.4f}")
    print(f"   Train time: {train_time:.2f}s")


# ===== BUOC 3.4: SO SANH KET QUA =====
print(f"\n{'=' * 60}")
print("BUOC 3.4: So sanh ket qua")
print("=" * 60)

# Tao bang so sanh
print(f"\n{'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'Time':>8}")
print("-" * 55)
for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:>10.4f} {res['f1_score']:>10.4f} {res['train_time']:>7.2f}s")

# Tim model tot nhat (theo F1-score)
best_name = max(results, key=lambda x: results[x]["f1_score"])
best_res = results[best_name]
print(f"\n>> Model tot nhat: {best_name}")
print(f"   Accuracy: {best_res['accuracy']:.4f} | F1: {best_res['f1_score']:.4f}")


# ===== BUOC 3.5: CLASSIFICATION REPORT CHO MODEL TOT NHAT =====
print(f"\n--- Classification Report ({best_name}) ---")
print(classification_report(y_test, best_res["y_pred"],
                            target_names=["Negative", "Positive"]))


# ===== BUOC 3.6: VE BIEU DO =====
os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)

# --- Bieu do 1: So sanh Accuracy va F1 cua 4 models ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names = list(results.keys())
accuracies = [results[n]["accuracy"] for n in model_names]
f1_scores = [results[n]["f1_score"] for n in model_names]

# Bar chart: Accuracy
colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
bars1 = axes[0].bar(model_names, accuracies, color=colors)
axes[0].set_title("Model Comparison — Accuracy", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0.7, 1.0)  # Zoom in de thay ro su khac biet
for bar, acc in zip(bars1, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{acc:.4f}", ha='center', fontsize=11, fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)

# Bar chart: F1-Score
bars2 = axes[1].bar(model_names, f1_scores, color=colors)
axes[1].set_title("Model Comparison — F1-Score", fontsize=14, fontweight="bold")
axes[1].set_ylabel("F1-Score")
axes[1].set_ylim(0.7, 1.0)
for bar, f1 in zip(bars2, f1_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{f1:.4f}", ha='center', fontsize=11, fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, "results", "02_model_comparison.png"), dpi=150, bbox_inches="tight")
print(f"\n>> Bieu do so sanh -> results/02_model_comparison.png")
plt.close()

# --- Bieu do 2: Confusion Matrix cua model tot nhat ---
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best_res["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"], ax=ax)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, "results", "03_confusion_matrix.png"), dpi=150, bbox_inches="tight")
print(f">> Confusion matrix -> results/03_confusion_matrix.png")
plt.close()


# ===== BUOC 3.7: LUU MODEL + TF-IDF VECTORIZER =====
os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)

# Luu model tot nhat
# joblib.dump() = luu object Python thanh file .pkl
# Sau nay chi can joblib.load() la dung lai duoc, khong can train lai
joblib.dump(best_res["model"], os.path.join(ROOT_DIR, "models", "best_model.pkl"))
print(f"\n>> Da luu model: models/best_model.pkl ({best_name})")

# Luu TF-IDF vectorizer (QUAN TRONG!)
# Tai sao phai luu vectorizer?
# -> Khi predict review moi, can dung CUNG vocabulary da xay tu train data
# -> Neu khong luu, phai train lai tu dau moi lan predict
joblib.dump(tfidf, os.path.join(ROOT_DIR, "models", "tfidf_vectorizer.pkl"))
print(f">> Da luu vectorizer: models/tfidf_vectorizer.pkl")


# ===== KET QUA =====
print(f"\n{'=' * 60}")
print(f"BUOC 3 HOAN THANH!")
print(f"{'=' * 60}")
print(f"""
Tom tat:
  TF-IDF: {X_train.shape[1]:,} features (max_features=10000, ngram 1-2)
  
  Ket qua 4 models:""")
for name, res in results.items():
    marker = " << BEST" if name == best_name else ""
    print(f"    {name:<25} Acc={res['accuracy']:.4f}  F1={res['f1_score']:.4f}{marker}")
print(f"""
  Model tot nhat: {best_name} (F1={best_res['f1_score']:.4f})
  Da luu: models/best_model.pkl + models/tfidf_vectorizer.pkl
  Bieu do: results/02_model_comparison.png + 03_confusion_matrix.png

>> Buoc tiep theo: Chay file src/step4_predict.py de thu predict review moi
          hoac chay app.py de tao web demo
""")
