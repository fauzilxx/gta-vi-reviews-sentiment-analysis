# IMPLEMENTATION PLAN
## Klasifikasi Sentimen pada Ulasan Pengguna Grand Theft Auto VI Menggunakan Metode Naive Bayes, Logistic Regression, dan Support Vector Machine

### 1. Project Overview

Membangun aplikasi analisis sentimen terhadap ulasan pengguna game Grand Theft Auto VI (GTA VI) menggunakan pendekatan klasifikasi teks. Proyek ini bertujuan untuk membandingkan performa tiga algoritma klasifikasi, yaitu Multinomial Naive Bayes, Logistic Regression, dan Support Vector Machine (Linear SVM).

Seluruh algoritma akan menggunakan representasi fitur TF-IDF dengan dataset, preprocessing, serta pembagian data latih dan data uji yang sama agar perbandingan performanya adil.

Hasil eksperimen akan diintegrasikan ke dalam dashboard interaktif berbasis Streamlit yang menampilkan eksplorasi data, distribusi sentimen, hasil evaluasi model, perbandingan performa, dan prediksi sentimen ulasan baru.

### 2. Project Objectives

1. Melakukan eksplorasi dan analisis awal terhadap dataset ulasan pengguna GTA VI.
2. Menerapkan preprocessing teks untuk mempersiapkan data sebelum klasifikasi.
3. Mengubah teks ulasan menjadi representasi numerik menggunakan TF-IDF.
4. Mengimplementasikan tiga algoritma klasifikasi: Multinomial Naive Bayes, Logistic Regression, dan Linear SVM.
5. Membandingkan performa ketiga algoritma menggunakan metrik evaluasi klasifikasi.
6. Mengembangkan dashboard interaktif untuk menyajikan hasil analisis dan memungkinkan pengguna mencoba prediksi sentimen.

### 3. Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Text Processing:** NLTK atau library Python yang relevan
- **Feature Extraction:** Scikit-learn TF-IDF Vectorizer
- **Machine Learning:** Scikit-learn
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Dashboard:** Streamlit
- **Model Serialization:** Joblib

### 4. Implementation Phases

#### Phase 1 — Dataset Inspection & Exploratory Data Analysis

- Memuat dataset ulasan GTA VI dari Kaggle.
- Memeriksa struktur dataset, nama kolom, tipe data, dan jumlah observasi.
- Mengidentifikasi missing values dan data duplikat.
- Memeriksa distribusi rating, platform, bahasa, serta atribut lain yang tersedia.
- Memvisualisasikan distribusi data dan karakteristik ulasan.
- Mengidentifikasi sumber dan mekanisme pembentukan label sentimen.

**Output:** Dataset yang telah dipahami karakteristiknya dan hasil Exploratory Data Analysis (EDA).

#### Phase 2 — Data Preprocessing & Sentiment Labeling

- Menghapus data yang tidak relevan atau tidak memiliki informasi penting.
- Menghapus duplikasi dan menangani missing values.
- Melakukan pembersihan teks sesuai kebutuhan dataset.
- Melakukan case folding dan normalisasi teks apabila diperlukan.
- Melakukan tokenization.
- Menentukan label sentimen berdasarkan label yang tersedia atau rating pengguna dengan batas kategori yang dijelaskan secara metodologis.
- Memeriksa distribusi kelas sentimen untuk mengetahui potensi ketidakseimbangan data.

**Catatan:** Penentuan label harus mengikuti struktur dan skala rating aktual dataset. Jangan menetapkan ambang rating sebelum memeriksa data. Jika label dibentuk dari rating, kolom rating tidak boleh digunakan sebagai fitur prediksi karena dapat menyebabkan data leakage.

**Output:** Dataset bersih yang memiliki teks ulasan dan target sentimen.

#### Phase 3 — Feature Extraction & Dataset Splitting

- Membagi dataset menjadi data training dan testing menggunakan stratified split.
- Menggunakan random state yang konsisten agar eksperimen dapat direproduksi.
- Mengubah teks ulasan menjadi fitur numerik menggunakan TF-IDF.
- Melakukan fitting TF-IDF hanya pada data training, kemudian menerapkannya pada data testing.
- Menggunakan representasi fitur yang sama untuk ketiga algoritma.

**Output:** Data training dan testing dalam bentuk fitur TF-IDF.

#### Phase 4 — Model Development

Mengimplementasikan tiga algoritma klasifikasi:

1. **Multinomial Naive Bayes**
   - Menggunakan pendekatan probabilistik untuk mengklasifikasikan sentimen berdasarkan fitur teks.
   - Menjadi model pembanding yang sederhana dan efisien.

2. **Logistic Regression**
   - Menggunakan pendekatan regresi logistik untuk menentukan kelas sentimen.
   - Menjadi model pembanding berbasis klasifikasi linear.

3. **Support Vector Machine (Linear SVM)**
   - Menggunakan LinearSVC untuk membangun hyperplane pemisah antar kelas.
   - Menjadi model pembanding berbasis margin maksimum.

Seluruh model dilatih menggunakan data training dan fitur TF-IDF yang sama.

**Output:** Tiga model klasifikasi sentimen yang telah dilatih.

#### Phase 5 — Model Evaluation & Comparison

Melakukan evaluasi menggunakan data testing yang belum digunakan saat pelatihan.

Metrik evaluasi:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Langkah evaluasi:
- Menghasilkan prediksi dari setiap model.
- Menghitung metrik evaluasi untuk setiap algoritma.
- Menampilkan classification report.
- Membandingkan hasil ketiga model dalam tabel dan grafik.
- Menentukan model terbaik berdasarkan metrik yang relevan, terutama Macro F1-Score jika distribusi kelas tidak seimbang.
- Melakukan analisis kesalahan klasifikasi untuk memahami jenis ulasan yang sulit diprediksi.

**Output:** Hasil evaluasi dan perbandingan performa ketiga algoritma.

#### Phase 6 — Model Persistence

- Menyimpan model klasifikasi yang telah dilatih menggunakan Joblib.
- Menyimpan TF-IDF Vectorizer yang telah di-fit pada data training.
- Menyimpan informasi kelas dan konfigurasi model jika diperlukan.
- Memastikan model dan vectorizer dapat dimuat kembali tanpa melakukan pelatihan ulang saat dashboard dijalankan.

**Output:** File model dan vectorizer yang siap digunakan oleh dashboard.

#### Phase 7 — Streamlit Dashboard Development

Mengembangkan dashboard interaktif dengan beberapa halaman atau bagian utama:

**A. Overview**
- Deskripsi singkat proyek.
- Informasi dataset.
- Jumlah ulasan dan distribusi sentimen.
- Ringkasan hasil eksperimen.

**B. Exploratory Data Analysis**
- Distribusi sentimen.
- Distribusi rating jika tersedia.
- Distribusi ulasan berdasarkan platform jika tersedia.
- Visualisasi panjang ulasan.
- Kata atau istilah yang sering muncul.

**C. Model Comparison**
- Perbandingan Accuracy, Precision, Recall, dan F1-Score.
- Grafik perbandingan performa ketiga algoritma.
- Tabel hasil evaluasi.
- Penentuan model dengan performa terbaik.

**D. Model Evaluation**
- Confusion Matrix setiap model.
- Classification Report.
- Analisis hasil klasifikasi dan kesalahan prediksi.

**E. Sentiment Prediction**
- Input teks ulasan secara manual.
- Memilih model klasifikasi yang ingin digunakan.
- Melakukan preprocessing dan transformasi TF-IDF.
- Menampilkan hasil prediksi sentimen.

**Output:** Dashboard Streamlit yang dapat digunakan untuk mengeksplorasi data, melihat hasil eksperimen, dan mencoba prediksi sentimen.

#### Phase 8 — Testing & Finalization

- Memastikan seluruh proses preprocessing berjalan dengan benar.
- Memastikan ketiga model menggunakan dataset dan skema evaluasi yang sama.
- Menguji prediksi menggunakan beberapa contoh ulasan.
- Memastikan dashboard dapat dijalankan tanpa error.
- Memeriksa konsistensi hasil antara notebook eksperimen dan dashboard.
- Menyusun dokumentasi, hasil evaluasi, kesimpulan, serta keterbatasan penelitian.

**Output:** Implementasi proyek, dashboard, dan dokumentasi akhir.

### 5. Suggested Project Structure

```text
gta-vi-sentiment-analysis/
│
├── data/
│   ├── raw/
│   │   └── gta_vi_reviews.csv
│   └── processed/
│       └── processed_reviews.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── models/
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── linear_svm.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── dashboard/
│   ├── app.py
│   └── components.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

Struktur dapat disederhanakan selama proses pengembangan apabila diperlukan, tetapi pemisahan kode eksperimen, model, dan dashboard tetap perlu dijaga.

### 6. Implementation Priorities

Dengan keterbatasan waktu pengerjaan, prioritas implementasi adalah:

**Priority 1 — Core Data Mining**
- Dataset inspection.
- Preprocessing dan penentuan label.
- TF-IDF.
- Training tiga algoritma.
- Evaluasi dan perbandingan model.

**Priority 2 — Functional Dashboard**
- Overview.
- Visualisasi EDA.
- Perbandingan performa model.
- Confusion matrix.
- Prediksi sentimen ulasan baru.

**Priority 3 — Finalization**
- Pengujian.
- Perbaikan tampilan.
- Dokumentasi.
- Penyusunan kesimpulan dan keterbatasan.

Fitur tambahan seperti hyperparameter tuning mendalam, cross-validation, analisis aspek (aspect-based sentiment analysis), dan deteksi sarkasme tidak menjadi prioritas utama karena keterbatasan waktu.

### 7. Important Implementation Guidelines

1. Jangan mengasumsikan nama kolom atau format dataset sebelum memeriksa file CSV.
2. Jangan membuat label sentimen secara sembarang. Dokumentasikan dasar pembentukan label.
3. Hindari data leakage, khususnya saat melakukan TF-IDF dan membentuk target dari rating.
4. Gunakan preprocessing, pembagian data, dan fitur yang konsisten untuk ketiga model.
5. Simpan model dan vectorizer agar dashboard tidak perlu melakukan training ulang.
6. Pisahkan proses training dan dashboard agar aplikasi lebih ringan.
7. Jangan mengarang hasil evaluasi. Semua angka dan kesimpulan harus berasal dari eksperimen aktual.
8. Gunakan notebook sebagai tempat eksperimen dan dashboard sebagai tempat penyajian hasil.

### 8. Expected Final Deliverables

- Proposal proyek Data Mining.
- Dataset dan dokumentasi preprocessing.
- Notebook EDA dan eksperimen klasifikasi.
- Tiga model klasifikasi yang telah dilatih.
- Hasil evaluasi dan perbandingan algoritma.
- Dashboard interaktif berbasis Streamlit.
- README berisi deskripsi proyek dan cara menjalankan aplikasi.
- Kesimpulan mengenai performa model terbaik beserta keterbatasannya.


## AI CODING AGENT INSTRUCTIONS

Act as a Data Mining Engineer and Python Developer. Your task is to implement this project incrementally based on the implementation plan.

### Development Rules

1. **Inspect before implementing.**
   - Start by inspecting the existing project directory and dataset.
   - Identify the actual CSV filename, columns, data types, missing values, duplicates, rating scale, language, and available labels.
   - Do not assume the dataset schema.

2. **Validate the methodology.**
   - Determine whether sentiment labels already exist or need to be derived from ratings.
   - Explain the proposed labeling strategy before implementing it.
   - If the dataset has no reliable sentiment labels, stop and report the issue instead of inventing labels.
   - Avoid data leakage.

3. **Implement incrementally.**
   - Complete one phase at a time.
   - Keep code modular, readable, and maintainable.
   - Avoid unnecessary abstraction and overengineering.
   - Do not create unnecessary files or features.

4. **Model consistency.**
   - Use the same train/test split and TF-IDF features for all three classifiers.
   - Use a fixed random state and stratified splitting when appropriate.
   - Evaluate using Accuracy, Precision, Recall, Macro F1-Score, and Confusion Matrix.
   - Do not fabricate metrics or results.

5. **Dashboard requirements.**
   - Use Streamlit.
   - Keep the interface clean, responsive, and easy to understand.
   - Separate data exploration, model comparison, model evaluation, and sentiment prediction.
   - Load saved models and vectorizer instead of retraining models every time the dashboard starts.
   - Ensure prediction uses the exact preprocessing pipeline used during training.

6. **Verification.**
   - Run the scripts and test the application.
   - Fix errors before moving to the next phase.
   - Verify that dashboard metrics match the experiment results.
   - Update README and requirements.txt.

### First Task

Do not immediately build the entire application.

First, inspect the project directory and the provided GTA VI dataset. Then report:

- Dataset structure and available columns.
- Number of records.
- Missing values and duplicates.
- Available rating values and their distribution.
- Whether sentiment labels already exist.
- Potential data leakage risks.
- Recommended sentiment labeling strategy.
- Recommended implementation steps based on the actual dataset.

Wait for methodology confirmation before proceeding to model training.