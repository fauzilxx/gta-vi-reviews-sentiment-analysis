# Analisis sentimen ulasan GTA VI

Proyek membandingkan Multinomial Naive Bayes, Logistic Regression, dan Linear SVM
menggunakan fitur TF-IDF yang sama. Label adalah **kategori rating**, bukan anotasi
sentimen manual: 0–4 Negative, 5–7 Neutral, dan 8–10 Positive. Ambang ini melanjutkan
eksperimen yang sudah ada; belum ada validasi manusia atau kajian sensitivitas ambang.

## Dataset dan provenance

Sumber yang diberikan pemilik proyek:
[Kaggle: ektarr/gta-vi-user-reviews](https://www.kaggle.com/datasets/ektarr/gta-vi-user-reviews/data).
Dokumentasi halaman tidak dapat dibaca melalui alat browsing saat tindak lanjut audit.
Versi unduhan, lisensi, metode pengumpulan, dan arti `review_id` belum terverifikasi.
URL ini merupakan informasi dari pemilik proyek, bukan bukti bahwa file lokal identik
dengan versi Kaggle tertentu. Hash file lokal dicatat dalam laporan.

`reviews.csv` dan `data/raw/gta_vi_reviews.csv` masing-masing berisi 51.385 baris.
Perbedaan isinya terbatas pada satu ulasan: file raw memiliki tambahan 20 pemisah baris
Unicode U+2028. Jangan mengganti salah satunya tanpa mencatat alasan.

Dataset processed yang diwarisi berisi 49.981 ulasan. Pipeline lama menghapus 1.156
duplikasi ID, lalu 105 duplikasi teks, lalu 143 teks dengan kurang dari dua kata.
Penghapusan ID pertama perlu dikaji: ID berulang muncul lintas platform dan sebagian
mempunyai teks/rating berbeda. Dataset processed dipertahankan sampai makna ID jelas;
hasil saat ini hanya berlaku untuk subset tersebut.

## Keputusan setelah audit

- Artifact awal tetap di lokasi semula. Checksum pembanding ada di
  `reports/baseline_artifacts.json`.
- Eksperimen yang dipersiapkan untuk tahap model menggunakan
  `data/processed/grouped_v1/` dan `models/grouped_v1/`.
- Dataset processed tidak dibuat ulang. Ada alasan teknis untuk membuat ulang fitur:
  split lama memiliki 12 ulasan test dengan teks bersih yang juga ada di train.
- Split baru memakai `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
  Fold pertama dipilih sebelum training; setiap kelompok `clean_text` identik berada
  dalam satu split. Proporsi test mendekati 20%, tidak harus tepat 20% karena kelompok
  tidak boleh dipecah.
- Konflik label pada teks identik dipertahankan dan ditempatkan dalam satu kelompok.
  Label tidak ditimpa menggunakan mayoritas atau ditebak dari teks.
- TF-IDF tetap memakai unigram/bigram, maksimal 20.000 fitur, `min_df=2`,
  `max_df=0.95`, `sublinear_tf=True`, dan tanpa stopword removal. Fitting hanya pada train.
- Indeks dataset, urutan baris matriks, dan `review_id` disimpan di
  `split_membership.csv`. Konfigurasi, hash, dan versi paket dicatat di `manifest.json`.
- Vektor kosong dipertahankan dan didiagnosis di `reports/zero_vectors.csv`.
  Menghapusnya dari test akan membuat evaluasi mengabaikan ulasan yang sulit
  direpresentasikan. Metrik mendatang harus mencakup seluruh test dan melaporkan
  keterbatasan ini.

## Preprocessing dan multilingual

Artifact baseline dan `grouped_v1` memakai **`legacy-v1`**. Fungsi
`clean_text(text)` mempertahankan perilaku itu sehingga inferensi tetap sesuai data
training. Regex lama dapat menghilangkan combining marks, dan tokenisasi berbasis
kata/spasi belum memadai untuk semua bahasa.

`clean_text(text, version="unicode-v2")` menyediakan normalisasi NFC dan preservasi
combining marks. Versi ini sudah diuji pada contoh Thai dan aksen terurai, tetapi
**belum digunakan pada dataset atau vectorizer**. Mengaktifkannya untuk eksperimen
memerlukan dataset dan artifact versi baru; jangan mencampurkannya dengan vectorizer
`legacy-v1`. Perubahan ini juga belum menyelesaikan segmentasi bahasa tanpa spasi.

`assign_sentiment` kini menolak rating pecahan, nilai di luar 0–10, nilai nonfinite,
dan boolean. Pipeline preprocessing menolak menimpa file processed yang sudah ada.
Kebijakan deduplikasi ID di dalam pipeline lama belum diubah karena semantik ID belum
diketahui; jangan menjalankannya pada raw untuk menetapkan dataset baru sebelum isu
ini diputuskan.

## Menjalankan dan memverifikasi

Jalankan dari direktori root proyek dengan PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m src.audit
```

Perintah audit membaca data, memverifikasi artifact versi baru, memperbarui laporan,
dan membuat grafik EDA. Tidak melakukan preprocessing atau fitting. Notebook 01–03
menampilkan inspeksi dan validasi, tanpa otomatis menjalankan ulang pipeline.

Untuk membuat fitur ketika direktori versi tujuan **belum ada**:

```powershell
.\.venv\Scripts\python.exe -B -m src.feature_extraction
```

Perintah tersebut menolak menimpa artifact. Untuk eksperimen lain, panggil
`run_feature_extraction` dengan `processed_dir` dan `models_dir` versi baru.
Jika notebook dibuka, kernel harus memakai interpreter `.venv` proyek.

`requirements.txt` mencatat dependensi langsung dan mengunci versi paket inti
sesuai manifest model tersimpan. `requirements-lock.txt` merekam
paket terpasang saat tindak lanjut audit untuk environment Python/Windows ini;
belum diverifikasi lewat instalasi bersih pada platform lain.

## Hasil dan langkah berikutnya

Lihat `reports/dataset_audit.json`, grafik `reports/figures/dataset_overview.png`,
diagnostik ID/label, serta output notebook untuk angka yang berasal dari file aktual.
`reports/verification.json` merekam pemeriksaan hasil akhir.

Phase 1 memiliki notebook dan grafik EDA dasar. Phase 2 tetap menggunakan hasil
lama dengan keterbatasan metodologi yang dicatat. Phase 3 memiliki split berkelompok
dan verifikasi artifact. Phase 4–5 telah dijalankan pada artifact tersebut:
ketiga classifier tersimpan dengan Joblib, dan hasil evaluasi berasal dari prediksi
aktual terhadap 9.996 ulasan test. Phase 6 menggunakan model/vectorizer tersimpan
dengan pemeriksaan integritas. Dashboard Phase 7 tersedia dan pengujian fungsional
Phase 8 dicatat pada `reports/dashboard_verification.json`.

## Menjalankan dashboard lokal

Dari root proyek pada PowerShell, gunakan environment yang sudah tersedia:

```powershell
Set-Location E:\gta-vi-reviews
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Buka **http://localhost:8501**. Hentikan server dengan **Ctrl+C**.
Dashboard tidak melakukan training, fitting TF-IDF, atau regenerasi artifact.
Model dan vectorizer dimuat sekali melalui cache; perubahan file membatalkan cache
agar integritas diperiksa ulang. Path data/model ditentukan dari lokasi source,
bukan direktori kerja saat aplikasi dijalankan.

Untuk menyiapkan environment baru, gunakan Python **3.13.0** dan pasang dependensi:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Jangan menimpa `.venv` yang sudah digunakan hanya untuk mencoba instalasi ulang.
Artifact sklearn menggunakan versi paket dan Python yang dicatat dalam manifest;
dashboard menolak environment yang tidak cocok. Instalasi bersih pada mesin lain
belum diuji. Streamlit 1.67.0 dan Altair 6.3.0 sudah tersedia pada environment proyek.

Lima halaman dashboard:

1. **Overview:** tujuan, jumlah data, distribusi sentimen, informasi sumber dan label.
2. **Exploratory data analysis:** rating, platform, sentimen, seluruh rentang panjang
   teks, dan frekuensi token tanpa stopword removal. EDA memakai seluruh processed
   dataset untuk deskripsi, tanpa mengubah vocabulary atau eksperimen.
3. **Model comparison:** lima metrik, grafik bersumbu 0–100%, dan penanda terbaik
   berdasarkan Macro F1. Precision/Recall pada tabel utama adalah macro average.
4. **Model evaluation:** pilihan ketiga model, confusion matrix counts/normalisasi
   per baris, classification report dan support tiap kelas. Semua test tetap dihitung.
5. **Sentiment prediction:** satu ulasan baru, pilihan model, dan label hasil prediksi.
   Input kosong, hasil cleaning kosong, serta vektor tanpa fitur yang dikenali ditolak
   dengan pesan penjelasan. Tidak ada confidence/probabilitas buatan untuk LinearSVC.

Logika pemuatan, validasi, ringkasan data, dan prediksi berada di `src/predict.py`.
Cache UI ada di `dashboard/data.py`; komponen grafik di `dashboard/components.py`;
halaman ada di `dashboard/app_pages/`. Preprocessing inferensi memakai `legacy-v1`
sesuai manifest, bukan `unicode-v2`. Label merupakan proksi rating, bukan anotasi
sentimen manusia; hasil dapat keliru, terutama pada kelas Neutral dan bahasa yang
kurang terwakili.

Pengujian lokal:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m src.evaluate --verify-only
```

Tes dashboard menggunakan `streamlit.testing.v1.AppTest`, termasuk semua halaman,
pilihan model, normalisasi, kecocokan metrik dengan eksperimen, prediksi terhadap
baris test tersimpan, error handling, serta larangan pemanggilan fit. Data sintetis
pada tes pipeline berada di direktori sementara; artifact proyek tidak dibuat ulang.

Untuk pemeriksaan pada server nyata, gunakan server lokal terpisah:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.headless=true --server.address=127.0.0.1 --server.port=8765 --server.enableAgentApi=true
```

Lalu, dari terminal kedua:

```powershell
.\.venv\Scripts\python.exe -B scripts\check_dashboard_live.py
```

Flag Agent API hanya dipakai untuk pengujian lokal dan tidak disimpan dalam konfigurasi.
Hasil pemeriksaan ada di `reports/dashboard_live_checks.json`. Hentikan server uji
dengan Ctrl+C setelah selesai. Pengujian visual browser dan type checking memiliki
batas environment yang dirinci di `reports/dashboard_verification.json`.

## Training dan evaluasi Phase 4–5

`PRD.md` menjadi acuan sesuai instruksi pengguna karena `AGENTS.md` belum tersedia.
PRD tidak menetapkan hyperparameter numerik. Konfigurasi baseline dipilih sebelum
evaluasi dan dicatat di `reports/grouped_v1/model_plan.json`:

- MultinomialNB: `alpha=1`, `fit_prior=True` (prior empiris training).
- LogisticRegression: `C=1`, `class_weight='balanced'`, `solver='lbfgs'`,
  `max_iter=2000`, `tol=1e-4`, `random_state=42`.
- LinearSVC: `C=1`, `class_weight='balanced'`, `dual='auto'`, `max_iter=10000`,
  `tol=1e-4`, `random_state=42`.

Ketiga model memakai matriks training yang sama, tanpa split atau fitting TF-IDF
baru. Perbandingan ini membandingkan konfigurasi tersebut, termasuk perbedaan
kebijakan pembobotan; tidak mengisolasi pengaruh algoritma saja. Tidak ada tuning
atau pemilihan hyperparameter menggunakan test.

Laporan lengkap: [evaluation_report.md](reports/grouped_v1/evaluation_report.md).
Metrik agregat ada di `metrics.csv`, metrik per kelas di `per_class_metrics.csv`,
prediksi setiap baris di `predictions.csv`, serta identitas kesalahan di
`misclassifications.csv`, semuanya dalam `reports/grouped_v1/`. Confusion matrix
tersedia dalam CSV dan grafik counts/normalisasi per baris. Notebook
`04_model_training.ipynb` dan `05_model_evaluation.ipynb` memuat output aktual.

Model dan metadata training disimpan dalam `models/grouped_v1/`. Pemuatan ulang
telah diverifikasi dengan membandingkan prediksi sebelum/sesudah serialisasi pada
training, lalu mencocokkan prediksi ulang test, metrik, dan confusion matrix terhadap
hasil tersimpan. Buktinya ada di `reports/grouped_v1/evaluation_verification.json`.

Perintah dari root proyek:

```powershell
.\.venv\Scripts\python.exe -B -m src.train
.\.venv\Scripts\python.exe -B -m src.evaluate
```

Untuk memeriksa hasil yang sudah ada tanpa menulis hasil atau melakukan training:

```powershell
.\.venv\Scripts\python.exe -B -m src.evaluate --verify-only
```

Validator memeriksa classification report JSON terpisah dan gabungan beserta
agregatnya, metrik CSV, confusion matrix, cakupan kelas/model, jumlah test, metrik
subgroup vektor kosong/nonkosong, baseline mayoritas, dan daftar salah klasifikasi.
Prediksi dihitung ulang dari model tersimpan. Angka nonfinite dan struktur laporan
yang hilang/berlebih ditolak. Parameter model juga dicocokkan dengan rencana training.

Penguatan validator setelah audit tidak mengubah model atau hasil evaluasi awal.
`reports/grouped_v1/validator_v2_verification.json` mencatat pemeriksaan terbaru.
Hash source dalam `evaluation.json` tetap merekam kode ketika hasil awal dibuat;
hash validator terkini dicatat terpisah pada bukti verifikasi terbaru tersebut.

Perintah training dan pembuatan evaluasi **menolak menimpa hasil yang sudah ada**. Gunakan opsi
`--models-dir` dan/atau `--reports-dir` versi baru untuk eksperimen yang memang
diminta; direktori fitur tetap dapat menunjuk artifact Phase 3 yang sama.
Jangan mengulang eksperimen hanya untuk mencoba meningkatkan skor test yang sudah
terlihat. Jika tuning kelak diminta, gunakan validasi dari training dan jelaskan
bahwa test ini sudah digunakan untuk evaluasi baseline.

Skill proyek tersedia di `.agents/skills/gta-vi-sentiment-analysis/SKILL.md`.
Salinan sumber tetap tersedia di `skill-drafts/gta-vi-sentiment-analysis/SKILL.md`.
Checksum pemasangan dicatat di `reports/skill_validation.json`.

Sebelum menetapkan dataset final, lengkapi versi/lisensi/proses pengumpulan Kaggle,
jelaskan `review_id`, dan tinjau label rating melalui anotasi manusia. Jangan
menganggap URL sumber sebagai konfirmasi keaslian seluruh ulasan.

`reports/label_review_sample.csv` menyediakan 60 contoh training (20 per kelas)
untuk ditinjau. Kolom `human_sentiment` dan `review_notes` sengaja kosong; belum ada
anotasi manusia. Perintah audit memperbarui sampel dari data training, jadi simpan
hasil anotasi yang sudah diisi di file terpisah agar tidak tertimpa.

Dashboard dibangun setelah pengguna menyetujui kelanjutan Phase 6–8. Keterbatasan
kelas Neutral tetap berlaku dan tidak diperbaiki dengan tuning terhadap test.
Pilihan konfigurasi eksperimen berikutnya hanya memakai train/validasi;
test akhir yang sudah dievaluasi tidak untuk tuning.
