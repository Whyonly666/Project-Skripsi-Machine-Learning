# ==========================================================
# KODE INI DIJALANKAN DI GOOGLE COLAB
# Tujuan: Melatih SVM 4 kernel x 5 rasio splitting (20 model),
# lalu menyimpan setiap model sebagai file .pkl terpisah
# dengan pola nama: svm_{kernel}_{train}_{test}.pkl
# Contoh: svm_linear_70_30.pkl, svm_rbf_80_20.pkl, dst.
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------------
# 1. Load dataset yang sudah melalui preprocessing & labeling
#    Kolom yang dibutuhkan minimal:
#    - 'stopword removal'  -> hasil akhir preprocessing
#    - 'Sentiment'         -> label 'Positif' / 'Negatif'
# ----------------------------------------------------------
df = pd.read_csv('Hasil_Labelling_Data_2Class.csv')  # ganti sesuai nama file Anda

df_clean = df.dropna(subset=['stopword removal'])

# ----------------------------------------------------------
# 2. Label encoding & Pembobotan fitur TF-IDF
#    Vectorizer di-fit SEKALI pada seluruh data, lalu dipakai
#    konsisten untuk semua skenario splitting di bawah ini.
#
#    Catatan: LabelEncoder mengurutkan label secara alfabetis,
#    sehingga 'Negatif' -> 0 dan 'Positif' -> 1.
#    Urutan ini SAMA dengan label_map di app.py:
#    {1: "Positif", 0: "Negatif"} -> tidak perlu diubah.
# ----------------------------------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_clean['Sentiment'])

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_clean['stopword removal'])

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print('Label mapping:', label_mapping)

joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print('Vectorizer TF-IDF tersimpan -> tfidf_vectorizer.pkl')

# ----------------------------------------------------------
# 3. Splitting data untuk 5 skenario rasio
# ----------------------------------------------------------
X_train70, X_test30, y_train70, y_test30 = train_test_split(X, y, test_size=0.30, random_state=42)
X_train75, X_test25, y_train75, y_test25 = train_test_split(X, y, test_size=0.25, random_state=42)
X_train80, X_test20, y_train80, y_test20 = train_test_split(X, y, test_size=0.20, random_state=42)
X_train85, X_test15, y_train85, y_test15 = train_test_split(X, y, test_size=0.15, random_state=42)
X_train90, X_test10, y_train90, y_test10 = train_test_split(X, y, test_size=0.10, random_state=42)


def tampilkan_distribusi(X_train, X_test, judul):
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    total_size = train_size + test_size

    labels = ['Data Training', 'Data Testing']
    values = [train_size, test_size]
    colors = ['skyblue', 'salmon']
    x_pos = np.arange(len(labels))

    plt.figure(figsize=(6, 4))
    bars = plt.bar(x_pos, values, color=colors, width=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                  height + 3,
                  f'{height} data\n({height / total_size * 100:.2f}%)',
                  ha='center', va='bottom', fontsize=10)

    plt.xticks(x_pos, labels)
    plt.title(f'Distribusi Data Latih dan Data Uji ({judul})', fontsize=12)
    plt.xlabel('Jenis Data')
    plt.ylabel('Jumlah Data')
    plt.ylim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.show()


tampilkan_distribusi(X_train70, X_test30, '70:30')
tampilkan_distribusi(X_train75, X_test25, '75:25')
tampilkan_distribusi(X_train80, X_test20, '80:20')
tampilkan_distribusi(X_train85, X_test15, '85:15')
tampilkan_distribusi(X_train90, X_test10, '90:10')

# ----------------------------------------------------------
# 4. Pelatihan 20 kombinasi model (4 kernel x 5 rasio)
#    Setiap model + skenario disimpan dalam dictionary
#    'semua_model' agar mudah diakses di langkah penyimpanan
# ----------------------------------------------------------
skenario = {
    '70_30': (X_train70, X_test30, y_train70, y_test30),
    '75_25': (X_train75, X_test25, y_train75, y_test25),
    '80_20': (X_train80, X_test20, y_train80, y_test20),
    '85_15': (X_train85, X_test15, y_train85, y_test15),
    '90_10': (X_train90, X_test10, y_train90, y_test10),
}

kernels = ['linear', 'rbf', 'poly', 'sigmoid']

semua_model = {}       # menyimpan objek model: semua_model['linear_70_30'] = svm_linear70
hasil_evaluasi = []     # menyimpan ringkasan metrik untuk CSV

for rasio_label, (X_train, X_test, y_train, y_test) in skenario.items():
    for k in kernels:
        key = f'{k}_{rasio_label}'
        print(f'Melatih SVM {k} ({rasio_label.replace("_", ":")}) ...')

        model = SVC(kernel=k, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        akurasi = accuracy_score(y_test, y_pred)
        presisi = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f'  -> Akurasi: {akurasi:.4f} | Presisi: {presisi:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')

        semua_model[key] = model

        train_label, test_label = rasio_label.split('_')
        hasil_evaluasi.append({
            'kernel': k,
            'rasio': f'{train_label}:{test_label}',
            'akurasi': round(akurasi, 4),
            'presisi': round(presisi, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
        })

# ----------------------------------------------------------
# 5. Tampilkan tabel ringkasan per skenario rasio (opsional,
#    sesuai format df_results_70, df_results_75, dst yang
#    sudah Anda buat)
# ----------------------------------------------------------
df_hasil_lengkap = pd.DataFrame(hasil_evaluasi)

for rasio_label in ['70:30', '75:25', '80:20', '85:15', '90:10']:
    print(f'\n=== Hasil Evaluasi Rasio {rasio_label} ===')
    print(df_hasil_lengkap[df_hasil_lengkap['rasio'] == rasio_label]
          .set_index('kernel')[['akurasi', 'presisi', 'recall', 'f1_score']])

# ----------------------------------------------------------
# 6. Visualisasi perbandingan 4 kernel per rasio
# ----------------------------------------------------------
def plot_perbandingan(rasio_label):
    subset = df_hasil_lengkap[df_hasil_lengkap['rasio'] == rasio_label]
    metrics = ['akurasi', 'presisi', 'recall', 'f1_score']
    metric_labels = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']

    x = np.arange(len(metrics))
    width = 0.2
    warna = {'linear': 'orange', 'rbf': 'red', 'poly': 'purple', 'sigmoid': 'green'}

    plt.figure(figsize=(10, 6))
    for i, k in enumerate(kernels):
        nilai = subset[subset['kernel'] == k][metrics].values.flatten()
        bars = plt.bar(x + (i - 1.5) * width, nilai, width, label=f'SVM {k.capitalize()}', color=warna[k])
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f'{bar.get_height():.2f}', ha='center', fontsize=8)

    plt.xticks(x, metric_labels)
    plt.ylabel('Skor')
    plt.xlabel('Metrik Evaluasi')
    plt.title(f'Perbandingan Performa Model SVM ({rasio_label})')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


for rasio_label in ['70:30', '75:25', '80:20', '85:15', '90:10']:
    plot_perbandingan(rasio_label)

# ----------------------------------------------------------
# 7. Simpan 20 model sebagai file .pkl terpisah
#    Pola nama: svm_{kernel}_{train}_{test}.pkl
#    (harus persis sama dengan yang dicari oleh app.py)
# ----------------------------------------------------------
for key, model in semua_model.items():
    nama_file = f'svm_{key}.pkl'   # contoh: svm_linear_70_30.pkl
    joblib.dump(model, nama_file)
    print(f'Tersimpan -> {nama_file}')

# ----------------------------------------------------------
# 8. Simpan ringkasan seluruh hasil evaluasi ke CSV
#    File ini dipakai Streamlit untuk menampilkan metrik
#    tanpa perlu menghitung ulang setiap kali dijalankan
# ----------------------------------------------------------
df_hasil_lengkap.to_csv('ringkasan_evaluasi_model.csv', index=False)
print('\nRingkasan evaluasi tersimpan -> ringkasan_evaluasi_model.csv')
print(df_hasil_lengkap)

# ----------------------------------------------------------
# 9. Unduh seluruh file ke komputer lokal
#    (jalankan di cell terpisah setelah training selesai)
# ----------------------------------------------------------
from google.colab import files

files.download('tfidf_vectorizer.pkl')
files.download('ringkasan_evaluasi_model.csv')

for key in semua_model.keys():
    files.download(f'svm_{key}.pkl')

print('\nSelesai! Total file yang diunduh: 1 vectorizer + 1 ringkasan CSV + 20 model .pkl')