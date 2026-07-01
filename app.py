import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing import full_preprocessing, load_kamus_normalisasi

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Aplikasi Pluang",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fix whitespace atas — harus dipanggil sebelum elemen apapun
st.markdown("""
<style>
/* Hapus whitespace kosong di atas halaman */
div[data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; }
.main .block-container { padding-top: 0 !important; margin-top: 0 !important; max-width: 100% !important; }
div.block-container { padding-top: 0 !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CSS GLOBAL
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
 
html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Inter', sans-serif !important;
}
 
/* Hide default Streamlit elements & remove ALL top whitespace */
#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
 
/* Hapus semua padding/margin atas yang menyebabkan whitespace */
.main .block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}
.main { padding: 0 !important; margin: 0 !important; }
[data-testid="stAppViewContainer"] { padding: 0 !important; margin: 0 !important; }
[data-testid="stAppViewContainer"] > section { padding: 0 !important; margin: 0 !important; }
div[data-testid="stVerticalBlock"] > div:first-child { margin-top: 0 !important; padding-top: 0 !important; }
 
/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    position: sticky; top: 0; z-index: 999;
    background: rgba(255,255,255,0.97) !important;
    backdrop-filter: blur(10px);
    border-bottom: 2px solid #e5e7eb !important;
    padding: 0 32px !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 16px 28px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
    color: #059669 !important;
    border-bottom: 2px solid #059669 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }
div[data-testid="stTabContent"] { padding: 0 !important; }
 
/* Button */
.stButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-size: 15px !important;
    font-weight: 600 !important; padding: 12px 28px !important;
    transition: all 0.2s !important; letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(5,150,105,0.35) !important;
}
 
/* Selectbox */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border-color: #d1d5db !important;
    font-size: 14px !important;
}
 
/* Textarea */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 1.5px solid #d1d5db !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
}
.stTextArea textarea:focus {
    border-color: #059669 !important;
    box-shadow: 0 0 0 3px rgba(5,150,105,0.15) !important;
}
 
/* Metric cards */
div[data-testid="stMetricValue"] {
    font-size: 22px !important; font-weight: 700 !important; color: #111827 !important;
}
div[data-testid="stMetricLabel"] { font-size: 12px !important; color: #6b7280 !important; }
 
/* Expander */
.streamlit-expanderHeader {
    font-weight: 600 !important; font-size: 14px !important;
    color: #374151 !important; border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONFIGURASI MODEL
# ============================================================
KERNEL_OPTIONS = {"Linear": "linear", "RBF": "rbf", "Polynomial": "poly", "Sigmoid": "sigmoid"}
RATIO_OPTIONS  = {"70:30": ("70","30"), "75:25": ("75","25"), "80:20": ("80","20"),
                  "85:15": ("85","15"), "90:10": ("90","10")}
WARNA          = {'linear':'#f59e0b','rbf':'#ef4444','poly':'#8b5cf6','sigmoid':'#10b981'}
LABEL_K        = {'linear':'SVM Linear','rbf':'SVM RBF','poly':'SVM Polynomial','sigmoid':'SVM Sigmoid'}

def nama_file_model(kv, rv):
    tr, te = RATIO_OPTIONS[rv]
    return f"svm_{kv}_{tr}_{te}.pkl"

# ============================================================
# LOAD RESOURCE
# ============================================================
@st.cache_resource
def load_vectorizer():
    return joblib.load('tfidf_vectorizer.pkl')

@st.cache_resource
def load_model_cached(nf):
    return joblib.load(nf) if os.path.exists(nf) else None

@st.cache_resource
def load_kamus():
    return load_kamus_normalisasi()

@st.cache_data
def load_ringkasan():
    if not os.path.exists('ringkasan_evaluasi_model.csv'):
        return None
    df = pd.read_csv('ringkasan_evaluasi_model.csv')
    for c in ['akurasi','presisi','recall','f1_score']:
        if c in df.columns and df[c].max() <= 1.0:
            df[c] = (df[c] * 100).round(2)
    return df

vectorizer        = load_vectorizer()
kamus_normalisasi = load_kamus()
df_ringkasan      = load_ringkasan()

# ============================================================
# LOGO HEADER ATAS
# ============================================================
import base64

def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Panggil sebelum st.markdown navbar
logo_b64 = get_image_base64("logo_pluang.png")

st.markdown(f"""
<div style="background:linear-gradient(135deg,#059669 0%,#10b981 100%);
    padding:14px 32px;display:flex;align-items:center;gap:12px;">
  <img src="data:image/png;base64,{logo_b64}"
       style="width:36px;height:36px;border-radius:8px;object-fit:cover;">
  <span style="color:white;font-size:16px;font-weight:700;letter-spacing:0.3px;">
    Analisis Sentimen Ulasan Aplikasi Pluang
  </span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGASI TAB UTAMA
# ============================================================
tab_beranda, tab_sistem, tab_visualisasi, tab_analisis = st.tabs([
    "Beranda", "Sistem", "Visualisasi", "Analisis Sentimen"
])

# ══════════════════════════════════════════════════════════════
#  TAB 1 — BERANDA
# ══════════════════════════════════════════════════════════════
import base64

def get_image_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

logo_b64 = get_image_base64("logo_pluang.png")
logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:100px;height:100px;border-radius:20px;object-fit:cover;box-shadow:0 4px 16px rgba(0,0,0,0.15);">' if logo_b64 else '<div style="font-size:64px;">📱</div>'

with tab_beranda:
    # Hero
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);
        padding:72px 64px;display:grid;grid-template-columns:1fr 1fr;gap:64px;
        align-items:center;">
      <div>
        <div style="display:inline-block;background:#d1fae5;color:#065f46;
            font-size:12px;font-weight:700;padding:5px 14px;border-radius:999px;
            letter-spacing:0.8px;margin-bottom:16px;">MACHINE LEARNING · NLP · SVM</div>
        <h1 style="font-size:38px;font-weight:800;line-height:1.2;color:#111827;margin:0 0 20px;">
          Analisis Sentimen<br>Ulasan Aplikasi Pluang<br>
          <span style="color:#059669;">Pada Google Play Store</span>
          <br>Menggunakan Model<br>
          <span style="color:;">Machine Learning</span>
        </h1>
        <p style="font-size:16px;color:#4b5563;line-height:1.75;margin:0 0 28px;">
          Sistem ini mengklasifikasikan sentimen ulasan pengguna aplikasi Pluang
          secara otomatis menggunakan algoritma <strong>Support Vector Machine (SVM)</strong>
          dengan empat jenis kernel untuk memperoleh perbandingan performa terbaik.
        </p>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <div style="background:white;border:1px solid #a7f3d0;border-radius:12px;
              padding:14px 20px;text-align:center;min-width:100px;">
            <div style="font-size:24px;font-weight:800;color:#059669;">7.179</div>
            <div style="font-size:12px;color:#6b7280;font-weight:500;">Scraping Data</div>
          </div>
          <div style="background:white;border:1px solid #a7f3d0;border-radius:12px;
              padding:14px 20px;text-align:center;min-width:100px;">
            <div style="font-size:24px;font-weight:800;color:#059669;">4</div>
            <div style="font-size:12px;color:#6b7280;font-weight:500;">Kernel SVM</div>
          </div>
          <div style="background:white;border:1px solid #a7f3d0;border-radius:12px;
              padding:14px 20px;text-align:center;min-width:100px;">
            <div style="font-size:24px;font-weight:800;color:#059669;">5</div>
            <div style="font-size:12px;color:#6b7280;font-weight:500;">Skenario Splitting</div>
          </div>
          <div style="background:;border:;border-radius:;
              padding:14px 20px;text-align:center;min-width:100px;">
            <div style="font-size:24px;font-weight:800;color:#059669;"></div>
            <div style="font-size:12px;color:#6b7280;font-weight:500;"></div>
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:center;justify-content:center;">
        <div style="width:100%;max-width:420px;border-radius:24px;overflow:hidden;
            box-shadow:0 20px 60px rgba(5,150,105,0.2);">
          <div style="background:linear-gradient(135deg,#059669,#10b981);
              padding:24px;text-align:center;">
            <div style="display:flex;align-items:center;justify-content:center;margin-bottom:8px;">
              {logo_html}
            </div>
            <div style="color:white;font-size:18px;font-weight:700;margin-top:8px;">
              Aplikasi Pluang
            </div>
            <div style="color:rgba(255,255,255,0.8);font-size:13px;margin-top:4px;">
              Siapa pun dan di mana pun, kamu berhak mendapat kesempatan yang sama untuk membangun kesejahteraan finansial. 
              Pluang hadir untuk menyederhanakan cara mengelola uang lewat fitur dan diversifikasi praktis.
            </div>
          </div>
          <div style="background:white;padding:20px 24px;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
              <div style="width:40px;height:40px;background:#f0fdf4;border-radius:10px;
                  display:flex;align-items:center;justify-content:center;font-size:20px;">💰</div>
              <div>
                <div style="font-size:13px;font-weight:600;color:#111827;">Reksa Dana & Emas Digital</div>
                <div style="font-size:12px;color:#6b7280;">Investasi mulai dari Rp10.000</div>
              </div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
              <div style="width:40px;height:40px;background:#f0fdf4;border-radius:10px;
                  display:flex;align-items:center;justify-content:center;font-size:20px;">📈</div>
              <div>
                <div style="font-size:13px;font-weight:600;color:#111827;">Saham AS & Kripto</div>
                <div style="font-size:12px;color:#6b7280;">Akses pasar global</div>
              </div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="width:40px;height:40px;background:#f0fdf4;border-radius:10px;
                  display:flex;align-items:center;justify-content:center;font-size:20px;">🔒</div>
              <div>
                <div style="font-size:13px;font-weight:600;color:#111827;">Terdaftar & Diawasi OJK</div>
                <div style="font-size:12px;color:#6b7280;">Aman dan terpercaya</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Tentang Pluang
    st.markdown("""
    <div style="max-width:1000px;margin:0 auto;padding:64px 64px 32px;">
      <div style="text-align:center;margin-bottom:40px;">
        <div style="display:inline-block;background:#d1fae5;color:#065f46;
            font-size:12px;font-weight:700;padding:5px 14px;border-radius:999px;
            letter-spacing:0.8px;margin-bottom:12px;">TENTANG APLIKASI</div>
        <h2 style="font-size:28px;font-weight:800;color:#111827;margin:0;">Apa itu Pluang?</h2>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:16px;padding:24px;">
          <div style="font-size:28px;margin-bottom:12px;">🏆</div>
          <h3 style="font-size:16px;font-weight:700;color:#111827;margin:0 0 8px;">Platform Investasi Terkemuka</h3>
          <p style="font-size:14px;color:#4b5563;line-height:1.65;margin:0;">
            Pluang merupakan platform investasi digital multiaset terkemuka di Indonesia yang
            menyediakan akses ke berbagai instrumen keuangan dalam satu ekosistem terintegrasi.
          </p>
        </div>
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:16px;padding:24px;">
          <div style="font-size:28px;margin-bottom:12px;">📊</div>
          <h3 style="font-size:16px;font-weight:700;color:#111827;margin:0 0 8px;">Beragam Instrumen Investasi</h3>
          <p style="font-size:14px;color:#4b5563;line-height:1.65;margin:0;">
            Menyediakan reksa dana, emas digital 24 karat, saham Amerika Serikat (S&P 500),
            serta aset kripto pilihan yang dapat diakses dengan modal terjangkau.
          </p>
        </div>
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:16px;padding:24px;">
          <div style="font-size:28px;margin-bottom:12px;">🌟</div>
          <h3 style="font-size:16px;font-weight:700;color:#111827;margin:0 0 8px;">Jutaan Pengguna Aktif</h3>
          <p style="font-size:14px;color:#4b5563;line-height:1.65;margin:0;">
            Dengan jutaan pengguna aktif di Indonesia, Pluang menghasilkan ribuan ulasan pada
            Google Play Store yang menjadi sumber data berharga untuk analisis sentimen.
          </p>
        </div>
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:16px;padding:24px;">
          <div style="font-size:28px;margin-bottom:12px;">🎯</div>
          <h3 style="font-size:16px;font-weight:700;color:#111827;margin:0 0 8px;">Tujuan Penelitian</h3>
          <p style="font-size:14px;color:#4b5563;line-height:1.65;margin:0;">
            Menganalisis sentimen ulasan pengguna secara otomatis menggunakan SVM multi-kernel
            untuk memahami persepsi pengguna terhadap layanan Pluang secara efisien dan akurat.
          </p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 2 — SISTEM
# ══════════════════════════════════════════════════════════════
with tab_sistem:
    st.markdown("""
    <div style="max-width:900px;margin:0 auto;padding:64px 48px;">
      <div style="text-align:center;margin-bottom:56px;">
        <div style="display:inline-block;background:#d1fae5;color:#065f46;
            font-size:12px;font-weight:700;padding:5px 14px;border-radius:999px;
            letter-spacing:0.8px;margin-bottom:12px;">ALUR PENELITIAN</div>
        <h2 style="font-size:28px;font-weight:800;color:#111827;margin:0 0 12px;">
          Tahapan Proses Sistem
        </h2>
        <p style="font-size:15px;color:#6b7280;max-width:540px;margin:0 auto;line-height:1.7;">
          Sistem analisis sentimen ini dibangun melalui delapan tahapan yang sistematis
          dan terstruktur dari pengumpulan data hingga deployment.
        </p>
      </div>
    """, unsafe_allow_html=True)

    proses = [
        {
            "no": "", "icon": "1", "judul": "Scraping Data",
            "warna_bg": "#f0fdf4", "warna_border": "#a7f3d0", "warna_no": "#059669",
            "deskripsi": "Data ulasan pengguna dikumpulkan secara otomatis dari halaman aplikasi Pluang pada Google Play Store menggunakan library <strong>google-play-scraper</strong> berbasis Python. Proses scraping menghasilkan <strong>7.179 ulasan</strong> dengan rentang waktu pengumpulan dari 1 Januari 2025 hingga 28 Mei 2026.",
            "tag": ["google-play-scraper", "Python", "7.179 Ulasan"]
        },
        {
            "no": "", "icon": "2", "judul": "Preprocessing Data",
            "warna_bg": "#fff7ed", "warna_border": "#fed7aa", "warna_no": "#ea580c",
            "deskripsi": "Tahapan pembersihan dan standarisasi data teks mentah menjadi representasi yang bersih dan siap diproses oleh model machine learning.",
            "tag": ["Cleaning", "Case Folding", "Spelling Normalization", "Tokenizing", "Stopword Removal"]
        },
        {
            "no": "", "icon": "3", "judul": "Pelabelan Sentimen (Labeling)",
            "warna_bg": "#fdf4ff", "warna_border": "#e9d5ff", "warna_no": "#9333ea",
            "deskripsi": "Pelabelan sentimen dilakukan berdasarkan nilai <strong>score/rating</strong> yang diberikan pengguna. Ulasan dengan score 1–2 dikategorikan sebagai <strong>Negatif</strong>, sedangkan ulasan dengan score 3–5 dikategorikan sebagai <strong>Positif</strong>.",
            "tag": ["😊 Positif (Score 3–5)", "😞 Negatif (Score 1–2)"]
        },
        {
            "no": "", "icon": "4", "judul": "Pembobotan Fitur TF-IDF",
            "warna_bg": "#eff6ff", "warna_border": "#bfdbfe", "warna_no": "#2563eb",
            "deskripsi": "Data teks yang telah diproses diubah menjadi representasi numerik menggunakan metode <strong>Term Frequency-Inverse Document Frequency (TF-IDF)</strong> dengan parameter <code>max_features=5000</code>, sehingga setiap kata memperoleh bobot yang mencerminkan tingkat kepentingannya dalam dokumen.",
            "tag": ["TF-IDF", "max_features=5000", "Representasi Numerik"]
        },
        {
            "no": "", "icon": "5", "judul": "Splitting Data",
            "warna_bg": "#f0fdfa", "warna_border": "#99f6e4", "warna_no": "#0d9488",
            "deskripsi": "Dataset dibagi menjadi data latih dan data uji menggunakan <strong>metode Hold-Out</strong> dengan lima skenario rasio pembagian yang berbeda untuk mengevaluasi pengaruh proporsi data terhadap performa model.",
            "tag": ["70:30", "75:25", "80:20", "85:15", "90:10"]
        },
        {
            "no": "", "icon": "6", "judul": "Model Klasifikasi SVM",
            "warna_bg": "#fff1f2", "warna_border": "#fecdd3", "warna_no": "#e11d48",
            "deskripsi": "Klasifikasi sentimen dilakukan menggunakan algoritma <strong>Support Vector Machine (SVM)</strong> dengan pendekatan multi-kernel. Setiap kernel memiliki karakteristik berbeda dalam membentuk batas keputusan (decision boundary) untuk memisahkan kelas sentimen.",
            "tag": ["Kernel Linear", "Kernel RBF", "Kernel Polynomial", "Kernel Sigmoid"]
        },
        {
            "no": "", "icon": "7", "judul": "Evaluasi Model",
            "warna_bg": "#fefce8", "warna_border": "#fde68a", "warna_no": "#d97706",
            "deskripsi": "Kinerja setiap model dievaluasi menggunakan empat metrik utama yang diturunkan dari <strong>confusion matrix</strong>, yaitu akurasi, presisi, recall, dan F1-score. Evaluasi dilakukan pada seluruh kombinasi kernel dan rasio splitting untuk menentukan model terbaik.",
            "tag": ["Akurasi", "Presisi", "Recall", "F1-Score", "Confusion Matrix"]
        },
        {
            "no": "", "icon": "8", "judul": "Deployment (Website)",
            "warna_bg": "#f0fdf4", "warna_border": "#a7f3d0", "warna_no": "#059669",
            "deskripsi": "Model terbaik diimplementasikan ke dalam aplikasi web interaktif menggunakan framework <strong>Streamlit</strong> berbasis Python. Pengguna dapat memilih kernel SVM dan rasio splitting yang diinginkan, memasukkan teks ulasan, dan memperoleh hasil klasifikasi sentimen secara langsung.",
            "tag": ["Streamlit", "Python", "Web Application", "Interaktif"]
        },
    ]

    for i, p in enumerate(proses):
        tag_html = "".join([
            f'<span style="background:white;border:1px solid {p["warna_border"]};'
            f'color:{p["warna_no"]};font-size:12px;font-weight:600;'
            f'padding:4px 12px;border-radius:999px;">{t}</span>'
            for t in p["tag"]
        ])
        arrow = "" if i == len(proses)-1 else f"""
        <div style="text-align:center;padding:8px 0;font-size:22px;color:#059669;">↓</div>
        """
        st.markdown(f"""
        <div style="background:{p['warna_bg']};border:1.5px solid {p['warna_border']};
            border-radius:18px;padding:24px 28px;margin-bottom:0;">
          <div style="display:flex;align-items:flex-start;gap:16px;">
            <div style="min-width:48px;height:48px;background:{p['warna_no']};
                border-radius:12px;display:flex;align-items:center;justify-content:center;
                font-size:22px;">{p['icon']}</div>
            <div style="flex:1;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <span style="font-size:11px;font-weight:800;color:{p['warna_no']};
                    letter-spacing:1px;">{p['no']}</span>
                <h3 style="font-size:16px;font-weight:700;color:#111827;margin:0;">
                  {p['judul']}
                </h3>
              </div>
              <p style="font-size:14px;color:#4b5563;line-height:1.65;margin:0 0 12px;">
                {p['deskripsi']}
              </p>
              <div style="display:flex;flex-wrap:wrap;gap:8px;">{tag_html}</div>
            </div>
          </div>
        </div>
        {arrow}
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 3 — VISUALISASI
# ══════════════════════════════════════════════════════════════
with tab_visualisasi:
    st.markdown("""
    <div style="padding:48px 64px 0;">
      <div style="text-align:center;margin-bottom:36px;">
        <div style="display:inline-block;background:#d1fae5;color:#065f46;
            font-size:12px;font-weight:700;padding:5px 14px;border-radius:999px;
            letter-spacing:0.8px;margin-bottom:12px;">HASIL EVALUASI MODEL</div>
        <h2 style="font-size:28px;font-weight:800;color:#111827;margin:0 0 10px;">
          Visualisasi & Evaluasi Model
        </h2>
        <p style="font-size:15px;color:#6b7280;max-width:500px;margin:0 auto;">
          Pilih kernel SVM dan skenario splitting data untuk melihat
          perbandingan performa model secara interaktif.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter controls
    with st.container():
        st.markdown('<div style="padding:0 64px;">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 2, 6])
        with c1:
            kernel_vis_label = st.selectbox(
                " Kernel SVM:",
                ["Semua Kernel"] + list(KERNEL_OPTIONS.keys()),
                key="vis_kernel"
            )
        with c2:
            rasio_vis = st.selectbox(
                " Rasio Splitting:",
                list(RATIO_OPTIONS.keys()),
                index=2,
                key="vis_rasio"
            )
        st.markdown('</div>', unsafe_allow_html=True)

    if df_ringkasan is not None:
        subset = df_ringkasan[df_ringkasan['rasio'] == rasio_vis].copy()
        if kernel_vis_label != "Semua Kernel":
            kv = KERNEL_OPTIONS[kernel_vis_label]
            subset = subset[subset['kernel'] == kv]

        if not subset.empty:
            # Bar chart
            metrics       = ['akurasi','presisi','recall','f1_score']
            metric_labels = ['Akurasi','Presisi','Recall','F1-Score']
            kernels_plot  = subset['kernel'].tolist()
            n             = len(kernels_plot)
            x             = np.arange(len(metrics))
            width         = min(0.18, 0.7/n)

            fig, ax = plt.subplots(figsize=(11, 6))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f9fafb')

            for i, k in enumerate(kernels_plot):
                row    = subset[subset['kernel'] == k].iloc[0]
                vals   = [row[m] for m in metrics]
                offset = (i - n/2 + 0.5) * (width + 0.025)
                bars   = ax.bar(x + offset, vals, width,
                                label=LABEL_K.get(k,k),
                                color=WARNA.get(k,'#059669'),
                                alpha=0.88, zorder=3,
                                edgecolor='white', linewidth=0.8)
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.4,
                            f'{bar.get_height():.2f}',
                            ha='center', va='bottom',
                            fontsize=8, fontweight='600', color='#374151')

            ax.set_xticks(x)
            ax.set_xticklabels(metric_labels, fontsize=12, fontweight='600')
            ax.set_ylabel('Skor (%)', fontsize=11, color='#6b7280')
            ax.set_xlabel('Metrik Evaluasi', fontsize=11, color='#6b7280')
            ax.set_ylim(0, 118)
            ax.tick_params(colors='#6b7280')
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0, color='#d1d5db')
            ax.set_axisbelow(True)
            ax.legend(loc='upper right', framealpha=0.95, fontsize=10,
                      edgecolor='#e5e7eb', fancybox=True)
            ax.set_title(f'Perbandingan Performa Model SVM ({rasio_vis})',
                         fontsize=14, fontweight='800', color='#111827', pad=20)
            fig.tight_layout(pad=2)

            st.markdown('<div style="padding:20px 64px 0;">', unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # Tabel evaluasi
        st.markdown("""
        <div style="padding:32px 64px 0;">
          <h3 style="font-size:17px;font-weight:700;color:#111827;margin:0 0 16px;">
            📋 Tabel Hasil Evaluasi
          </h3>
        </div>
        """, unsafe_allow_html=True)

        rows_html = ""
        for _, row in subset.iterrows():
            is_best = row['akurasi'] == subset['akurasi'].max()
            bg = "#f0fdf4" if is_best else "white"
            badge = ' <span style="background:#059669;color:white;font-size:10px;font-weight:700;padding:2px 8px;border-radius:999px;margin-left:6px;">TERBAIK</span>' if is_best else ""
            rows_html += f"""
            <tr style="background:{bg};border-top:1px solid #f3f4f6;">
              <td style="padding:14px 20px;font-weight:600;color:#111827;">
                {LABEL_K.get(row['kernel'], row['kernel'])}{badge}
              </td>
              <td style="padding:14px 20px;text-align:center;font-weight:{'700' if is_best else '400'};
                  color:{'#059669' if is_best else '#374151'};">{row['akurasi']:.2f}%</td>
              <td style="padding:14px 20px;text-align:center;color:#374151;">{row['presisi']:.2f}%</td>
              <td style="padding:14px 20px;text-align:center;color:#374151;">{row['recall']:.2f}%</td>
              <td style="padding:14px 20px;text-align:center;color:#374151;">{row['f1_score']:.2f}%</td>
            </tr>"""

        st.markdown(f"""
        <div style="padding:0 64px 64px;">
          <div style="border-radius:16px;overflow:hidden;border:1px solid #e5e7eb;
              box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <table style="width:100%;border-collapse:collapse;font-size:14px;">
              <thead>
                <tr style="background:#f0fdf4;">
                  <th style="padding:14px 20px;text-align:left;font-weight:700;
                      color:#374151;font-size:13px;">Kernel</th>
                  <th style="padding:14px 20px;text-align:center;font-weight:700;
                      color:#374151;font-size:13px;">Akurasi</th>
                  <th style="padding:14px 20px;text-align:center;font-weight:700;
                      color:#374151;font-size:13px;">Presisi</th>
                  <th style="padding:14px 20px;text-align:center;font-weight:700;
                      color:#374151;font-size:13px;">Recall</th>
                  <th style="padding:14px 20px;text-align:center;font-weight:700;
                      color:#374151;font-size:13px;">F1-Score</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Expander semua 20 model
        with st.expander("📊 Lihat perbandingan seluruh 20 kombinasi model"):
            df_all = df_ringkasan.copy()
            df_all.columns = [c.replace('_',' ').title() for c in df_all.columns]
            for c in ['Akurasi','Presisi','Recall','F1 Score']:
                if c in df_all.columns:
                    df_all[c] = df_all[c].apply(lambda v: f"{v:.2f}%")
            st.dataframe(df_all, use_container_width=True, hide_index=True)
    else:
        st.info("📂 File `ringkasan_evaluasi_model.csv` belum tersedia. Jalankan `colab_save_model.py` terlebih dahulu.")

# ══════════════════════════════════════════════════════════════
#  TAB 4 — ANALISIS SENTIMEN
# ══════════════════════════════════════════════════════════════
with tab_analisis:
    st.markdown("""
    <div style="padding:48px 64px 0;">
      <div style="text-align:center;margin-bottom:40px;">
        <div style="display:inline-block;background:#d1fae5;color:#065f46;
            font-size:12px;font-weight:700;padding:5px 14px;border-radius:999px;
            letter-spacing:0.8px;margin-bottom:12px;">COBA SISTEM</div>
        <h2 style="font-size:28px;font-weight:800;color:#111827;margin:0 0 10px;">
          Analisis Sentimen Ulasan
        </h2>
        <p style="font-size:15px;color:#6b7280;max-width:480px;margin:0 auto;">
          Masukkan ulasan aplikasi Pluang secara bebas dan sistem akan
          mengklasifikasikan sentimennya secara otomatis.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Contoh ulasan
    st.markdown("""
    <div style="padding:0 64px 24px;">
      <p style="font-size:13px;font-weight:600;color:#374151;margin:0 0 12px;">
        💡 Contoh Ulasan — Klik untuk menggunakannya:
      </p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:700px;">
        <div style="border:2px solid #a7f3d0;border-radius:14px;padding:16px;background:#ecfdf5;">
          <p style="font-weight:700;font-size:13px;color:#059669;margin:0 0 4px;">😊 Ulasan Positif</p>
          <p style="font-size:13px;font-style:italic;color:#374151;margin:0;">
            "Aplikasi ini sangat membantu saya belajar investasi dari nol, fiturnya lengkap dan mudah dipahami."
          </p>
        </div>
        <div style="border:2px solid #fecaca;border-radius:14px;padding:16px;background:#fef2f2;">
          <p style="font-weight:700;font-size:13px;color:#dc2626;margin:0 0 4px;">😞 Ulasan Negatif</p>
          <p style="font-size:13px;font-style:italic;color:#374151;margin:0;">
            "Aplikasi sering error dan lambat, sudah lama tidak bisa login padahal saldo masih ada."
          </p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Form input
    with st.container():
        st.markdown('<div style="padding:0 64px;">', unsafe_allow_html=True)

        col_k, col_r = st.columns(2)
        with col_k:
            pred_kernel = st.selectbox("Pilih Kernel SVM:", list(KERNEL_OPTIONS.keys()), key="pred_k")
        with col_r:
            pred_rasio  = st.selectbox("Pilih Rasio Splitting:", list(RATIO_OPTIONS.keys()), index=2, key="pred_r")

        user_input = st.text_area(
            "Tulis Ulasan Anda:",
            placeholder="Contoh: Aplikasi Pluang sangat membantu untuk investasi...",
            height=130, key="review_input"
        )

        c_btn1, c_btn2, c_btn3 = st.columns([3, 3, 2])
        with c_btn2:
            klik = st.button("🔍 Analisis Sentimen Sekarang", key="btn_analisis")

        st.markdown('</div>', unsafe_allow_html=True)

    # Hasil prediksi
    if klik:
        st.markdown('<div style="padding:0 64px 64px;">', unsafe_allow_html=True)
        if not user_input.strip():
            st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            kv    = KERNEL_OPTIONS[pred_kernel]
            nf    = nama_file_model(kv, pred_rasio)
            model = load_model_cached(nf)

            if model is None:
                st.error(f"❌ File model `{nf}` tidak ditemukan. Pastikan semua file .pkl tersedia di folder yang sama.")
            else:
                with st.spinner("🔄 Memproses dan menganalisis teks ulasan..."):
                    clean  = full_preprocessing(user_input, kamus_normalisasi)
                    vektor = vectorizer.transform([clean])
                    pred   = model.predict(vektor)[0]
                    label  = {1:"Positif", 0:"Negatif"}.get(pred, str(pred))

                # Kotak hasil — posisi naik ke atas, margin atas dikurangi
                _, col_hasil, _ = st.columns([2, 3, 2])
                with col_hasil:
                    if label == "Positif":
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#ecfdf5,#d1fae5);
                            border:2px solid #6ee7b7;border-radius:16px;padding:20px 24px;
                            text-align:center;margin-top:4px;
                            box-shadow:0 4px 16px rgba(5,150,105,0.15);">
                          <div style="font-size:36px;margin-bottom:6px;">😊</div>
                          <p style="font-size:11px;font-weight:700;color:#065f46;
                              letter-spacing:1px;margin:0 0 4px;">HASIL ANALISIS SENTIMEN</p>
                          <p style="font-size:26px;font-weight:800;color:#059669;margin:0 0 6px;">
                            Positif
                          </p>
                          <p style="font-size:12px;color:#065f46;margin:0;">
                            Model: <strong>SVM {pred_kernel}</strong> &nbsp;|&nbsp;
                            Rasio: <strong>{pred_rasio}</strong>
                          </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#fef2f2,#fee2e2);
                            border:2px solid #fca5a5;border-radius:16px;padding:20px 24px;
                            text-align:center;margin-top:4px;
                            box-shadow:0 4px 16px rgba(220,38,38,0.12);">
                          <div style="font-size:36px;margin-bottom:6px;">😞</div>
                          <p style="font-size:11px;font-weight:700;color:#991b1b;
                              letter-spacing:1px;margin:0 0 4px;">HASIL ANALISIS SENTIMEN</p>
                          <p style="font-size:26px;font-weight:800;color:#dc2626;margin:0 0 6px;">
                            Negatif
                          </p>
                          <p style="font-size:12px;color:#991b1b;margin:0;">
                            Model: <strong>SVM {pred_kernel}</strong> &nbsp;|&nbsp;
                            Rasio: <strong>{pred_rasio}</strong>
                          </p>
                        </div>
                        """, unsafe_allow_html=True)

                # Spacer antara kotak hasil dan expander
                st.markdown('<div style="margin-top:32px;"></div>', unsafe_allow_html=True)

                with st.expander("🔎 Detail preprocessing teks"):
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.markdown("**📝 Teks asli:**")
                        st.info(user_input)
                    with col_p2:
                        st.markdown("**✅ Setelah preprocessing:**")
                        st.success(clean if clean.strip() else "(teks kosong setelah diproses)")

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""

</div>
""", unsafe_allow_html=True)