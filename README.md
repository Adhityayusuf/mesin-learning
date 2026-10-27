# Mesin Learning

Repositori praktikum **Mesin Learning** (Machine Learning) Semester 5 —
Politeknik Negeri Malang (Polinema).

Berisi **29 notebook Jupyter** yang dikerjakan di **Google Colab**,
dikelompokkan per jobsheet (`JS02`–`JS07`) plus satu notebook **Kuis 1**.
Setiap notebook diawali badge **Open in Colab** — klik badge tersebut
untuk langsung membukanya di Colab tanpa perlu install apa pun.

## Daftar Isi

- [Struktur Repositori](#struktur-repositori)
- [Detail per Jobsheet](#detail-per-jobsheet)
- [Library yang Dipakai](#library-yang-dipakai)
- [Cara Menjalankan](#cara-menjalankan)
- [Dataset yang Dibutuhkan](#dataset-yang-dibutuhkan)
- [Konteks](#konteks)

## Struktur Repositori

| Folder | Topik | Notebook |
|--------|-------|----------|
| `JS02/` | Dasar pengolahan data & citra | 4 praktikum + 1 tugas |
| `JS03/` | Preprocessing & representasi fitur | 3 praktikum + 1 tugas |
| `JS04/` | Regresi | 3 praktikum + 1 tugas |
| `JS05/` | Klasifikasi | 3 praktikum + 2 tugas |
| `JS06/` | Support Vector Machine (SVM) | 5 praktikum + 1 tugas |
| `JS07/` | Clustering | 3 praktikum + 1 tugas |
| root | Kuis 1 | 1 notebook |

## Detail per Jobsheet

### JS02 — Dasar Pengolahan Data dan Citra

| Notebook | Isi |
|----------|-----|
| `JS02_01.ipynb` | EDA dataset Titanic: inspeksi data, statistik deskriptif, histogram, boxplot, heatmap korelasi |
| `JS02_02.ipynb` | Menangani missing value Titanic (mean, konstanta, modus) lalu menyimpan `Titanic-Dataset-Fixed.csv` |
| `JS02_03.ipynb` | Encoding kategori dengan `LabelEncoder` dan standardisasi fitur dengan `StandardScaler` |
| `JS02_04.ipynb` | Dasar citra digital dengan OpenCV: baca gambar, konversi BGR→RGB, resize, grayscale |
| `JS02_Tugas.ipynb` | Tugas: klasifikasi dataset WBC (`wbc.csv`) dengan encoding + scaling |

### JS03 — Preprocessing dan Representasi Fitur

| Notebook | Isi |
|----------|-----|
| `JS03_01.ipynb` | Pipeline preprocessing (imputer, one-hot, scaling), seleksi fitur `chi2`/`SelectKBest`, lalu Logistic Regression |
| `JS03_02.ipynb` | Pembobotan TF-IDF pada corpus teks dengan `TfidfVectorizer` |
| `JS03_03.ipynb` | Ekstraksi channel warna dan histogram citra menggunakan Pillow (PIL) |
| `JS03_Tugas.ipynb` | Tugas: seleksi fitur + Logistic Regression pada `wbc.csv` |

### JS04 — Regresi

| Notebook | Isi |
|----------|-----|
| `JS04_0.ipynb` | Pembagian data train/validasi/test (8:1:1) termasuk stratified split |
| `JS04_01.ipynb` | Regresi linear dengan `statsmodels` (OLS): EDA, garis regresi, analisis residual |
| `JS04_02.ipynb` | Support Vector Regression (SVR) pada `Posisi_gaji.csv` beserta evaluasi MSE |
| `JS04_Tugas.ipynb` | Tugas: perbandingan Linear Regression vs SVR pada `insurance.csv` |

### JS05 — Klasifikasi

| Notebook | Isi |
|----------|-----|
| `JS05_01.ipynb` | K-Nearest Neighbors (KNN) pada dataset Iris |
| `JS05_02.ipynb` | Naive Bayes pada data sintetis (`make_classification`) |
| `JS05_03.ipynb` | Klasifikasi spam: `CountVectorizer` + Multinomial Naive Bayes |
| `JS05_Tugas1.ipynb` | Tugas: KNN + `SelectKBest` pada `voice.csv` (cross-validation, confusion matrix) |
| `JS05_Tugas2.ipynb` | Tugas: perbandingan `CountVectorizer` vs TF-IDF pada `spam.csv` |

### JS06 — Support Vector Machine

| Notebook | Isi |
|----------|-----|
| `JS06_01.ipynb` | Konsep dasar SVM: hyperplane dan margin pada data 2D |
| `JS06_02.ipynb` | Data non-linear: kernel trick dan SVM kernel RBF (`make_circles`) |
| `JS06_03.ipynb` | Pengaruh nilai `C` terhadap margin dan support vector |
| `JS06_04.ipynb` | SVM + PCA + `GridSearchCV` pada dataset wajah LFW |
| `JS06_05.ipynb` | Klasifikasi citra dari `images.zip` (ekstraksi fitur + SVM) |
| `JS06_Tugas.ipynb` | Tugas: SVM + `GridSearchCV` pada `voice.csv` |

### JS07 — Clustering

| Notebook | Isi |
|----------|-----|
| `JS07_01.ipynb` | Clustering KMeans pada dataset Iris |
| `JS07_02.ipynb` | Perbandingan KMeans vs DBSCAN dengan `silhouette_score` |
| `JS07_03.ipynb` | Kuantisasi warna citra menjadi 16 warna dengan `MiniBatchKMeans` |
| `JS07_Tugas.ipynb` | Tugas: segmentasi pelanggan kartu kredit (`CC_Customer_Dataset.csv`) dengan KMeans & DBSCAN |

### Kuis 1

| Notebook | Isi |
|----------|-----|
| `KUIS1_244107020045_M_Adhitya_Yusuf_Al_Ayyubi.ipynb` | EDA + pra-pengolahan dataset Census Income: imputasi, visualisasi, encoding, analisis korelasi, dan preprocessing MNIST |

## Library yang Dipakai

- **Manipulasi data:** pandas, numpy
- **Machine Learning:** scikit-learn, statsmodels, TensorFlow
- **Visualisasi:** matplotlib, seaborn, mpl_toolkits
- **Teks:** `CountVectorizer`, `TfidfVectorizer` (scikit-learn)
- **Citra:** OpenCV (`cv2`), Pillow (`PIL`), SciPy
- **Dataset:** ucimlrepo (UCI Machine Learning Repository)
- **Interaktif:** ipywidgets

## Cara Menjalankan

### Lewat Google Colab (disarankan)

1. Buka folder jobsheet yang diinginkan di GitHub.
2. Klik notebook-nya, lalu klik badge **Open in Colab** di baris pertama.
3. Upload dataset yang dibutuhkan (lihat tabel di bawah) ke panel **Files**
   Colab sebelum menjalankan sel.

### Secara lokal

```bash
git clone https://github.com/Adhityayusuf/mesin-learning.git
cd mesin-learning

pip install jupyter pandas numpy scikit-learn statsmodels tensorflow \
            matplotlib seaborn opencv-python pillow scipy ipywidgets ucimlrepo

jupyter lab
```

Jalankan notebook dari dalam folder jobsheet-nya (mis. `JS05/`), karena
banyak notebook membaca dataset dengan path relatif.

## Dataset yang Dibutuhkan

Dataset **tidak disertakan** di repositori ini. Beberapa notebook
mengandalkan upload manual via `google.colab.files` atau file yang harus
sudah ada di direktori kerja.

| File | Dipakai di |
|------|------------|
| `Titanic-Dataset.csv` | JS02_01, JS02_02, JS03_01 |
| `Titanic-Dataset-Fixed.csv` | JS02_02 (output) → JS02_03 |
| `Titanic-Dataset-selected.csv` | JS04_0 |
| `wbc.csv` | JS02_Tugas, JS03_Tugas |
| `dataset.csv` | JS04_01 |
| `Posisi_gaji.csv` | JS04_02 |
| `insurance.csv` | JS04_Tugas |
| `iris.csv` | JS05_01, JS07_01 |
| `spam.csv` | JS05_03, JS05_Tugas2 |
| `voice.csv` | JS05_Tugas1, JS06_Tugas |
| `CC_Customer_Dataset.csv` | JS07_Tugas |
| `Lenna.png` | JS02_04, JS03_03 |
| `images.zip` | JS06_05 |

> **Catatan:** `JS02_02` menyimpan hasil pembersihan sebagai
> `Titanic-Dataset-Fixed.csv`, sedangkan `JS02_03` membacanya dengan nama
> `Titanic-Dataset-fixed.csv`. Karena filesystem Colab case-sensitive,
> samakan dulu nama filenya sebelum menjalankan `JS02_03`.

## Konteks

Praktikum Mesin Learning, Politeknik Negeri Malang (Polinema),
Semester 5 — NIM 244107020045, M. Adhitya Yusuf Al-Ayyubi.
