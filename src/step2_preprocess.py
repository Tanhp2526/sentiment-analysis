"""
=============================================================
BUOC 2: TIEN XU LY TEXT (PREPROCESSING)
=============================================================
File nay lam 4 viec:
1. Doc train.csv va test.csv (output cua step1)
2. Lam sach text: xoa HTML, lowercase, xoa ky tu dac biet
3. Xoa stopwords (tu khong mang y nghia: "the", "is", "and"...)
4. Luu data da xu ly thanh processed CSV

Tai sao can preprocessing?
-> Text tho rat "ban": HTML tags, emoji, viet hoa, ky tu dac biet...
-> May tinh khong phan biet "GOOD" va "good" -> can chuan hoa
-> Tu nhu "the", "is", "a" xuat hien o MOI review -> khong giup phan biet
   positive vs negative -> loai bo de model tap trung vao tu quan trong
=============================================================
"""

# ===== IMPORT THU VIEN =====
import pandas as pd
import re       # Regular Expression: tim va thay the pattern trong text
import nltk     # Natural Language Toolkit: xu ly ngon ngu tu nhien
from nltk.corpus import stopwords       # Danh sach tu "vo nghia" (the, is, a...)
from nltk.stem import PorterStemmer     # Cat duoi tu: "running" -> "run"
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Tai danh sach stopwords tu NLTK (chi can chay 1 lan)
# stopwords la nhung tu xuat hien qua nhieu nhung KHONG mang y nghia phan biet
# Vi du: "the", "is", "and", "a", "to", "in"...
# Cau "this is a good movie" -> sau khi xoa stopwords -> "good movie"
# -> Model chi can tap trung vao "good" va "movie" la du!
nltk.download('stopwords', quiet=True)

# Tao set stopwords tieng Anh (set de tra cuu nhanh hon list)
STOP_WORDS = set(stopwords.words('english'))

# Tao stemmer
# Stemmer cat duoi tu de dua ve dang goc:
#   "running" -> "run"
#   "played"  -> "play"  
#   "happily" -> "happili" (khong hoan hao nhung nhanh!)
# Tai sao dung? Vi "running" va "run" co cung y nghia
# -> gom lai thanh 1 tu giup model hoc tot hon
stemmer = PorterStemmer()


def clean_text(text):
    """
    Ham lam sach 1 doan text.
    
    Input:  "This movie was ABSOLUTELY AMAZING!!! <br/><br/>Best film ever!!! :)"
    Output: "movi absolut amaz best film ever"
    
    Cac buoc xu ly (theo thu tu):
    """
    
    # Buoc 1: Xoa HTML tags
    # IMDB reviews chua <br/>, <p>, <b>... tu web
    # re.sub() = tim pattern va thay the bang chuoi khac
    # '<.*?>' = bat dau bang '<', ket thuc bang '>', '.*?' la bat ky ky tu nao
    # Vi du: "<br/>Hello<b>World</b>" -> "HelloWorld"
    text = re.sub('<.*?>', ' ', text)
    
    # Buoc 2: Chuyen tat ca thanh chu thuong (lowercase)
    # Tai sao? Vi "Good", "GOOD", "good" deu cung 1 nghia
    # Neu khong lowercase, may tinh coi chung la 3 tu KHAC NHAU
    text = text.lower()
    
    # Buoc 3: Xoa ky tu dac biet, chi giu lai chu cai va so
    # '[^a-zA-Z]' = bat ky ky tu nao KHONG phai chu cai -> thay bang dau cach
    # Vi du: "good!!! movie..." -> "good    movie   " 
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Buoc 4: Tach tu (tokenize) bang cach split theo dau cach
    # .split() tu dong bo cac khoang trang thua
    # Vi du: "good    movie   " -> ["good", "movie"]
    words = text.split()
    
    # Buoc 5: Xoa stopwords + Stemming (lam cung 1 luc cho nhanh)
    # Duyet qua tung tu:
    #   - Neu tu NAM TRONG stop_words -> bo qua (khong lay)
    #   - Neu KHONG -> stemming roi them vao ket qua
    # Vi du: ["this", "is", "a", "good", "movie"]
    #   -> "this" la stopword -> bo
    #   -> "is" la stopword -> bo
    #   -> "a" la stopword -> bo
    #   -> "good" -> stem -> "good" (giu lai)
    #   -> "movie" -> stem -> "movi" (giu lai)
    #   -> Ket qua: ["good", "movi"]
    words = [stemmer.stem(word) for word in words if word not in STOP_WORDS]
    
    # Buoc 6: Ghep lai thanh 1 chuoi
    # ["good", "movi"] -> "good movi"
    return ' '.join(words)


# ===== DOC DATA =====
print("=" * 60)
print("BUOC 2: Tien xu ly text (Preprocessing)")
print("=" * 60)

train_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", "train.csv"))
test_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", "test.csv"))
print(f"\nDoc xong: Train={train_df.shape[0]:,} | Test={test_df.shape[0]:,}")


# ===== XEM TRUOC KHI XU LY =====
print(f"\n[TRUOC KHI XU LY]")
sample = train_df.iloc[0]["review"]
print(f"   Raw: \"{sample[:150]}...\"")
print(f"   Clean: \"{clean_text(sample)[:150]}...\"")


# ===== AP DUNG CHO TOAN BO DATA =====
# .apply(clean_text) = ap dung ham clean_text cho TUNG dong trong cot "review"
# Voi 40,000 reviews, buoc nay se mat vai phut
print(f"\nDang xu ly {train_df.shape[0]:,} train reviews...")
train_df["clean_review"] = train_df["review"].apply(clean_text)
print(f">> Train xong!")

print(f"Dang xu ly {test_df.shape[0]:,} test reviews...")
test_df["clean_review"] = test_df["review"].apply(clean_text)
print(f">> Test xong!")


# ===== XEM KET QUA =====
print(f"\n[SAU KHI XU LY] - 3 vi du:")
print("-" * 60)
for i in range(3):
    row = train_df.iloc[i]
    label = "POS" if row["label"] == 1 else "NEG"
    print(f"   [{label}] Raw:   \"{row['review'][:80]}...\"")
    print(f"         Clean: \"{row['clean_review'][:80]}...\"")
    print()


# ===== KIEM TRA: CO REVIEW NAO RONG SAU KHI XU LY KHONG? =====
# Sau khi xoa stopwords, co the co review chi con "" (rong)
# -> Can kiem tra va xu ly
empty_train = (train_df["clean_review"].str.strip() == "").sum()
empty_test = (test_df["clean_review"].str.strip() == "").sum()
print(f"[KIEM TRA REVIEW RONG]")
print(f"   Train: {empty_train} reviews rong")
print(f"   Test:  {empty_test} reviews rong")

if empty_train > 0 or empty_test > 0:
    # Xoa cac review rong (khong co gi de model hoc)
    train_df = train_df[train_df["clean_review"].str.strip() != ""].reset_index(drop=True)
    test_df = test_df[test_df["clean_review"].str.strip() != ""].reset_index(drop=True)
    print(f"   -> Da xoa review rong!")
    print(f"   Train con: {train_df.shape[0]:,} | Test con: {test_df.shape[0]:,}")
else:
    print(f"   -> Khong co review rong. OK!")


# ===== LUU DATA DA XU LY =====
os.makedirs(os.path.join(ROOT_DIR, "data", "processed"), exist_ok=True)

# Luu 2 cot: clean_review (text da xu ly) va label
train_df[["clean_review", "label"]].to_csv(
    os.path.join(ROOT_DIR, "data", "processed", "train_clean.csv"), index=False
)
test_df[["clean_review", "label"]].to_csv(
    os.path.join(ROOT_DIR, "data", "processed", "test_clean.csv"), index=False
)

print(f"\n>> Da luu data da xu ly:")
print(f"   -> data/processed/train_clean.csv ({train_df.shape[0]:,} reviews)")
print(f"   -> data/processed/test_clean.csv  ({test_df.shape[0]:,} reviews)")


# ===== KET QUA =====
print(f"\n{'=' * 60}")
print(f"BUOC 2 HOAN THANH!")
print(f"{'=' * 60}")
print(f"""
Tom tat preprocessing:
  1. Xoa HTML tags (<br/>, <p>...)
  2. Chuyen lowercase (GOOD -> good)
  3. Xoa ky tu dac biet (!@#$%...)
  4. Xoa stopwords (the, is, a, and...)
  5. Stemming (running -> run, played -> play)
  
  Train: {train_df.shape[0]:,} reviews
  Test:  {test_df.shape[0]:,} reviews
""")
