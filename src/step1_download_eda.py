"""
=============================================================
BUOC 1: DOC DATASET IMDB + KHAM PHA DU LIEU (EDA)
=============================================================
File nay lam 3 viec:
1. Doc IMDB Dataset.csv (50,000 reviews da tai san)
2. Kham pha du lieu: xem cau truc, phan bo, do dai review
3. Chia train/test va luu CSV

Tai sao can EDA?
-> Truoc khi train model, phai HIEU data truoc:
  - Data co bao nhieu mau?
  - Co can bang khong? (positive vs negative bang nhau?)
  - Review dai bao nhieu tu?
=============================================================
"""

# ===== IMPORT THU VIEN =====
# pandas: doc va xu ly bang du lieu (DataFrame)
import pandas as pd

# matplotlib + seaborn: ve bieu do
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn: chia data train/test
from sklearn.model_selection import train_test_split

import os

# Thu muc goc cua project (sentiment-analysis/)
# Tinh tu vi tri file hien tai: src/step1.py -> di len 1 cap -> sentiment-analysis/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ===== BUOC 1.1: DOC DATASET =====
print("=" * 60)
print("BUOC 1: Doc IMDB Movie Reviews Dataset")
print("=" * 60)

# Doc file CSV co san
# File nay ban da tai tu Kaggle, da dat vao thu muc data/raw/
# Cau truc: 2 cot
#   - "review": noi dung review (tieng Anh)
#   - "sentiment": nhan "positive" hoac "negative" (dang chu)
CSV_PATH = os.path.join(ROOT_DIR, "data", "raw", "IMDB Dataset.csv")

print(f"\nDang doc file: IMDB Dataset.csv")
df = pd.read_csv(CSV_PATH)
print(f">> Doc xong! {df.shape[0]:,} reviews")


# ===== BUOC 1.2: CHUYEN NHAN THANH SO =====
# Tai sao chuyen?
# -> Model ML chi hieu so (0 va 1), khong hieu chu ("positive", "negative")
# -> Quy uoc: positive = 1, negative = 0
#
# .map() ap dung 1 dict de chuyen doi gia tri:
#   "positive" -> 1
#   "negative" -> 0
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Kiem tra xem co gia tri nao khong chuyen duoc khong (NaN)
if df["label"].isna().sum() > 0:
    print(f"CANH BAO: Co {df['label'].isna().sum()} gia tri khong hop le!")
else:
    print(f">> Chuyen nhan thanh so OK! (positive=1, negative=0)")


# ===== BUOC 1.3: KHAM PHA DATA =====
print(f"\n{'=' * 60}")
print(f"BUOC 2: Kham pha du lieu (EDA)")
print(f"{'=' * 60}")

# --- Kich thuoc ---
print(f"\n[KICH THUOC]")
print(f"   Tong: {df.shape[0]:,} reviews x {df.shape[1]} cot")
print(f"   Cot:  {df.columns.tolist()}")

# --- Xem mau ---
print(f"\n[3 REVIEWS DAU TIEN]")
print("-" * 60)
for i in range(3):
    review = df.iloc[i]
    text_preview = review["review"][:120]
    label = "POSITIVE" if review["label"] == 1 else "NEGATIVE"
    print(f"   Review {i+1} ({label}):")
    print(f"   \"{text_preview}...\"")
    print()

# --- Phan bo nhan ---
label_counts = df["label"].value_counts()
print(f"[PHAN BO NHAN]")
print(f"   Positive (1): {label_counts.get(1, 0):,} reviews")
print(f"   Negative (0): {label_counts.get(0, 0):,} reviews")

ratio = label_counts.get(1, 0) / label_counts.get(0, 0)
if 0.8 <= ratio <= 1.2:
    print(f"   -> Ty le: {ratio:.2f} -> CAN BANG! (gan 1:1)")
else:
    print(f"   -> Ty le: {ratio:.2f} -> KHONG CAN BANG!")

# --- Do dai review ---
# len(x.split()) = tach cau bang dau cach roi dem so tu
df["word_count"] = df["review"].apply(lambda x: len(str(x).split()))

print(f"\n[THONG KE DO DAI REVIEW] (so tu)")
print(f"   Ngan nhat:   {df['word_count'].min()} tu")
print(f"   Dai nhat:    {df['word_count'].max()} tu")
print(f"   Trung binh:  {df['word_count'].mean():.0f} tu")
print(f"   Trung vi:    {df['word_count'].median():.0f} tu")


# ===== BUOC 1.4: VE BIEU DO =====
os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bieu do 1: Phan bo nhan
colors = ["#e74c3c", "#2ecc71"]
sns.countplot(data=df, x="label", hue="label", ax=axes[0], palette=colors, legend=False)
axes[0].set_title("Positive vs Negative Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Count")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Negative (0)", "Positive (1)"])

for bar in axes[0].patches:
    axes[0].annotate(f'{int(bar.get_height()):,}',
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

# Bieu do 2: Phan bo do dai review
sns.histplot(data=df, x="word_count", bins=50, ax=axes[1], color="#3498db")
axes[1].set_title("Review Length Distribution (word count)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Number of words")
axes[1].set_ylabel("Count")

mean_words = df["word_count"].mean()
axes[1].axvline(x=mean_words, color="red", linestyle="--", linewidth=2)
axes[1].text(mean_words + 20, axes[1].get_ylim()[1] * 0.9,
             f"Mean: {mean_words:.0f} words", color="red", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, "results", "01_eda_overview.png"), dpi=150, bbox_inches="tight")
print(f"\n>> Da luu bieu do -> results/01_eda_overview.png")
plt.close()


# ===== BUOC 1.5: CHIA TRAIN / TEST =====
# Tai sao chia?
# -> Train: day model hoc (80% data)
# -> Test: kiem tra model co tot khong (20% data)
# -> KHONG duoc dung test data de train! (giong nhu khong duoc xem de truoc khi thi)
#
# train_test_split() tu dong chia ngau nhien
# - test_size=0.2 -> 20% lam test, 80% lam train
# - random_state=42 -> co dinh ket qua (chay lai van ra giong nhau)
# - stratify=df["label"] -> dam bao ty le pos/neg GIONG NHAU trong ca train va test

print(f"\n{'=' * 60}")
print(f"BUOC 3: Chia Train / Test")
print(f"{'=' * 60}")

train_df, test_df = train_test_split(
    df,                     # data goc
    test_size=0.2,          # 20% cho test
    random_state=42,        # co dinh seed
    stratify=df["label"]    # giu ty le pos/neg can bang
)

print(f"\n   Train: {train_df.shape[0]:,} reviews")
print(f"   Test:  {test_df.shape[0]:,} reviews")

# Kiem tra stratify co dung khong
train_pos = (train_df["label"] == 1).sum()
train_neg = (train_df["label"] == 0).sum()
test_pos = (test_df["label"] == 1).sum()
test_neg = (test_df["label"] == 0).sum()
print(f"   Train: Pos={train_pos:,} | Neg={train_neg:,} (ty le: {train_pos/train_neg:.2f})")
print(f"   Test:  Pos={test_pos:,} | Neg={test_neg:,} (ty le: {test_pos/test_neg:.2f})")


# ===== BUOC 1.6: LUU DATA =====
os.makedirs(os.path.join(ROOT_DIR, "data", "raw"), exist_ok=True)

# Chi luu 2 cot can thiet: review (text) va label (so)
train_df[["review", "label"]].to_csv(os.path.join(ROOT_DIR, "data", "raw", "train.csv"), index=False)
test_df[["review", "label"]].to_csv(os.path.join(ROOT_DIR, "data", "raw", "test.csv"), index=False)

print(f"\n>> Da luu:")
print(f"   -> data/raw/train.csv ({train_df.shape[0]:,} reviews)")
print(f"   -> data/raw/test.csv  ({test_df.shape[0]:,} reviews)")


# ===== KET QUA =====
print(f"\n{'=' * 60}")
print(f"BUOC 1 HOAN THANH!")
print(f"{'=' * 60}")
print(f"""

