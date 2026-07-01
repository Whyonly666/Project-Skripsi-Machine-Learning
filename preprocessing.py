"""
preprocessing.py
-----------------
Modul preprocessing teks yang DIBUAT IDENTIK dengan pipeline
preprocessing yang digunakan saat training model di Google Colab.

Urutan proses (harus sama persis dengan notebook Colab):
    1. Cleaning      (hapus URL, username, emoji, simbol, angka)
    2. Case Folding  (huruf kecil semua)
    3. Spelling Normalization (kamus kata baku dari GitHub)
    4. Tokenizing    (split spasi)
    5. Stopword Removal (NLTK bahasa Indonesia)
    6. Filtering tambahan (hapus_kata)
    7. Perbaikan kata manual (kamus_tidak_baku)

Letakkan file ini satu folder dengan app.py
"""

import re
import nltk
import pandas as pd
import requests
from io import BytesIO
from nltk.corpus import stopwords

# ----------------------------------------------------------
# Download resource NLTK (sekali saja, akan di-skip jika sudah ada)
# ----------------------------------------------------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = stopwords.words('indonesian')

# ----------------------------------------------------------
# Kata tambahan yang dihapus setelah stopword removal
# (persis seperti di notebook Colab)
# ----------------------------------------------------------
hapus_kata = ['ya', 'the', 'i', 'sih', 'sok', 'for', 'nih', 'is', 'ok', 'gpt', 'a', 'oke']

# ----------------------------------------------------------
# Kamus perbaikan kata manual tahap akhir
# (persis seperti di notebook Colab — silakan tambahkan
#  entri baru di sini jika Anda menemukan kata lain yang
#  hasilnya kurang tepat, misal akibat pemotongan simbol)
# ----------------------------------------------------------
kamus_tidak_baku = {
    'apk': 'aplikasi',
    'app': 'aplikasi',
    'ngedit': 'edit',
    'bagu': 'bagus',
}

# ----------------------------------------------------------
# Kamus normalisasi kata baku (diunduh sekali dari GitHub,
# lalu di-cache dengan st.cache_resource di app.py)
# ----------------------------------------------------------
KAMUS_URL = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"


def load_kamus_normalisasi() -> dict:
    """Mengunduh dan membangun dictionary kamus kata tidak baku -> baku."""
    response = requests.get(KAMUS_URL)
    file_excel = BytesIO(response.content)
    kamus_data = pd.read_excel(file_excel)
    return dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))


# ============================================================
# 1. CLEANING
# ============================================================
def remove_URL(text):
    if text is not None and isinstance(text, str):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)
    return text


def remove_usernames(text):
    if text is not None and isinstance(text, str):
        return re.sub(r'@\w+', '', text)
    return text


def remove_emoji(text):
    if text is not None and isinstance(text, str):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F700-\U0001F77F"
            u"\U0001F780-\U0001F7FF"
            u"\U0001F800-\U0001F8FF"
            u"\U0001FA00-\U0001FA6F"
            u"\U0001FA70-\U0001FAFF"
            u"\U0001F004-\U0001F0CF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    return text


def remove_symbols(text):
    if text is not None and isinstance(text, str):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def remove_numbers(text):
    if text is not None and isinstance(text, str):
        return re.sub(r'\d', '', text)
    return text


def cleaning(text: str) -> str:
    """Pipeline cleaning lengkap, urutan sama seperti di Colab."""
    text = remove_URL(text)
    text = remove_usernames(text)
    text = remove_emoji(text)
    text = remove_symbols(text)
    text = remove_numbers(text)
    return text


# ============================================================
# 2. CASE FOLDING
# ============================================================
def case_folding(text: str) -> str:
    if isinstance(text, str):
        return text.lower()
    return text


# ============================================================
# 3. SPELLING NORMALIZATION
# ============================================================
def spelling_normalization(text: str, kamus_dict: dict) -> str:
    """Mengganti kata tidak baku menjadi kata baku berdasarkan kamus GitHub."""
    if not isinstance(text, str):
        return ''
    words = text.split()
    replaced_words = []
    for word in words:
        if word in kamus_dict:
            baku_word = kamus_dict[word]
            if isinstance(baku_word, str) and all(c.isalpha() for c in baku_word):
                replaced_words.append(baku_word)
            else:
                replaced_words.append(word)
        else:
            replaced_words.append(word)
    return ' '.join(replaced_words)


# ============================================================
# 4. TOKENIZING
# ============================================================
def tokenize(text: str) -> list:
    return text.split()


# ============================================================
# 5. STOPWORD REMOVAL
# ============================================================
def remove_stopwords(tokens: list) -> list:
    return [word for word in tokens if word not in stop_words]


# ============================================================
# 6 & 7. FILTERING TAMBAHAN + PERBAIKAN KATA MANUAL
# ============================================================
def filter_kata_tambahan(text: str) -> str:
    return ' '.join([kata for kata in text.split() if kata not in hapus_kata])


def perbaiki_kata(text: str) -> str:
    kata_list = text.split()
    hasil = [kamus_tidak_baku.get(kata, kata) for kata in kata_list]
    return ' '.join(hasil)


# ============================================================
# PIPELINE LENGKAP — dipanggil dari app.py
# ============================================================
def full_preprocessing(text: str, kamus_normalisasi: dict) -> str:
    """
    Pipeline preprocessing lengkap, urutan SAMA PERSIS dengan
    notebook Google Colab:
    cleaning -> case folding -> spelling normalization ->
    tokenizing -> stopword removal -> filtering tambahan ->
    perbaikan kata manual
    """
    text = cleaning(text)
    text = case_folding(text)
    text = spelling_normalization(text, kamus_normalisasi)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    text = ' '.join(tokens)
    text = filter_kata_tambahan(text)
    text = perbaiki_kata(text)
    return text
