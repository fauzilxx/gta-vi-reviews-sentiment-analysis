---
name: gta-vi-sentiment-analysis
description: Audit, implement, train, evaluate, and present this GTA VI repository's sentiment pipeline using its existing data, TF-IDF artifacts, classifiers, and Streamlit application. Use for project-specific data mining methodology and Python engineering work.
---

# Workflow analisis sentimen GTA VI

Bertindak sebagai Data Mining Engineer dan Python Engineer untuk repository ini.
Gunakan bahasa Indonesia untuk laporan progres, penjelasan teknis, dan kesimpulan.
Lanjutkan pekerjaan yang sudah terverifikasi tanpa mengulang tahap yang selesai.
Semua path di bawah relatif terhadap root repository yang berisi `PRD.md`.

## Acuan dan scope

- Baca instruksi `AGENTS.md` yang berlaku jika tersedia, `PRD.md`, dan bagian
  `README.md` yang relevan sebelum mengubah pipeline. Jika `AGENTS.md` tidak ada,
  katakan demikian. Pengguna proyek ini telah mengarahkan agar `PRD.md` digunakan
  sebagai penggantinya. Jangan mengarang isi dokumen atau konfigurasi model.
- Instruksi pengguna menentukan scope. Skill ini tidak memberikan izin untuk
  training ulang, regenerasi data, deployment, atau dashboard di luar tugas aktif.
  Hormati persyaratan persetujuan evaluasi sebelum dashboard bila ditetapkan pengguna.
- Periksa status melalui file dan output aktual. Bedakan kode tersedia, eksekusi
  berhasil, dan hasil yang sudah diverifikasi. Jangan menganggap laporan agent
  sebelumnya sebagai pengganti bukti.

## Navigasi repository

- `data/raw/gta_vi_reviews.csv` adalah input raw pipeline; `reviews.csv` juga ada.
  Jangan berasumsi keduanya identik. Periksa checksum/perbedaan bila relevan.
- `data/processed/processed_reviews.csv` adalah dataset processed yang diwarisi.
- `data/processed/grouped_v1/` memuat fitur, label, `split_membership.csv`, dan
  `manifest.json`; `models/grouped_v1/` memuat vectorizer dan classifier terkait.
  Periksa manifest/instruksi untuk menentukan versi aktif; jangan mengunci semua
  pekerjaan berikutnya ke versi ini bila pengguna beralih eksperimen.
- `src/artifacts.py` dan `src/audit.py`: validasi integritas dan inspeksi dataset.
- `src/preprocessing.py`, `src/feature_extraction.py`: tahap data dan representasi.
- `src/modeling.py`, `src/train.py`, `src/evaluate.py`: konfigurasi, training, evaluasi.
- `notebooks/`: inspeksi dan eksperimen. Baca source, execution count, dan output;
  jangan mengeksekusi seluruh notebook secara otomatis karena sel lama bisa fit ulang.
- `reports/` dan `reports/grouped_v1/`: provenance, prediksi, metrik, serta grafik.
  Gunakan modul yang sudah ada sebelum membuat pipeline atau logika duplikat.

## Integritas dan reproduksibilitas

Periksa keberadaan artifact, skema kolom, kelas, dimensi sparse matrix, urutan label
terhadap membership, checksum manifest, serta kompatibilitas versi environment.
Validasi boleh memakai `transform` di memori, tetapi tidak melakukan `fit` atau
menulis ulang fitur diam-diam. Fitur MultinomialNB harus finite dan nonnegative.

Simpan konfigurasi, seed, versi Python/paket, identitas dataset, checksum input,
membership, dan urutan kelas bersama hasil. Pertahankan artifact yang sudah selesai.
Regenerasi hanya bila ada alasan teknis dan sesuai scope; jelaskan alasannya dan
gunakan direktori versi baru. Jangan mengganti artifact lama secara tersembunyi.

Jika artifact tidak cocok, dependensi tidak kompatibel, label tidak dapat
dipertanggungjawabkan, atau model gagal konvergen, hentikan tahap yang bergantung
pada masalah itu dan laporkan bukti serta dampaknya. Jangan memperbaiki hasil
dengan mengubah data, konfigurasi, atau angka secara diam-diam.

## Preprocessing multilingual dan pelabelan

- Inspeksi CSV aktual. Skema yang diwarisi mencakup `platform`, `review_id`,
  `published_date`, `score`, `review_text`, kemudian `clean_text` dan `sentiment`.
  Periksa ulang skema pada versi dataset baru.
- Strategi yang disepakati: rating integer **0–4 Negative, 5–7 Neutral, 8–10 Positive**.
  Validasi integer/rentang sebelum konversi; jangan memotong rating pecahan.
  Jangan mengganti ambang atau menebak label dari teks tanpa perubahan metodologi
  yang diminta pengguna. Label ini proksi rating, bukan anotasi manusia.
- `score` tidak menjadi fitur prediksi. Identitas, platform, tanggal, serta target
  tidak dimasukkan ke TF-IDF; gunakan hanya kolom teks yang disepakati.
- Pertahankan negasi dan karakter multilingual. Hindari stopword removal bahasa
  Inggris atau stemming massal pada semua bahasa. Jangan mengklaim distribusi
  bahasa tanpa anotasi atau metode deteksi yang benar-benar dijalankan.
- Cleaning training dan inferensi harus sama. Artifact saat skill dibuat memakai
  `legacy-v1`. Opsi `unicode-v2` memakai NFC dan mempertahankan combining marks;
  jangan mencampurkannya dengan vectorizer lama. Preservasi Unicode belum
  menyelesaikan segmentasi bahasa tanpa spasi atau semua masalah multilingual.
- Periksa duplikasi ID, teks asli, serta teks hasil normalisasi secara terpisah.
  ID lintas platform belum tentu ulasan identik. Jangan menganggap `keep=first`
  benar tanpa memahami arti ID. Catat konflik rating/label dan penanganannya.

## TF-IDF dan pencegahan leakage

- Untuk melanjutkan training, muat matriks dan label tersimpan. Jangan membuat split
  atau fitting TF-IDF ulang bila artifact yang disepakati masih valid.
- Bila fitur baru memang diminta, kelompokkan teks bersih identik agar tidak
  melintasi train/test, pertahankan stratifikasi sedekat mungkin, dan bekukan seed
  serta fold sebelum melihat performa. Konflik label bukan alasan mengubah label
  otomatis; tempatkan kelompok bersama dan dokumentasikan keputusannya.
- Fit vectorizer hanya pada training, kemudian transform test. Parameter berasal
  dari manifest; jangan mengganti vocabulary, n-gram, atau filtering secara tersembunyi.
  Ketiga classifier memakai fitur, urutan baris, dan split yang sama.
- Simpan mapping baris matriks ke dataset. Dimensi yang sama belum membuktikan
  bahwa matriks dan label berasal dari split yang sama.
- Test hanya untuk evaluasi akhir; jangan dipakai untuk tuning, threshold,
  vocabulary, class weight, atau early stopping. Jika tuning diminta, gunakan
  validasi dari training. Untuk CV teks yang ketat, fit TF-IDF di dalam setiap fold
  training; jangan menyebut CV bebas leakage jika vectorizer sudah fit pada semua
  training sebelum fold dipisahkan.

## Training dan Joblib

Latih **MultinomialNB**, **LogisticRegression**, dan **LinearSVC** menggunakan data
training tersimpan. Gunakan konfigurasi dari instruksi/rencana yang sudah disepakati.
Jika PRD tidak menentukan hyperparameter, nyatakan baseline sebagai keputusan
implementasi, bekukan sebelum test, dan jangan menyebutnya konfigurasi optimal.

Perhatikan kelas Neutral yang kecil. Pembobotan hanya dihitung dari training.
LR/LinearSVC dapat memakai `class_weight='balanced'` sesuai rencana. NB tidak memiliki
parameter `class_weight`; dokumentasikan prior atau sample weight jika digunakan.
Jelaskan perbedaan kebijakan pembobotan ketika membandingkan hasil antar model.

Pisahkan training dari evaluasi/dashboard. Simpan model dengan Joblib, parameter
aktual, kelas, seed bila relevan, peringatan konvergensi, dan hash manifest fitur.
Muat kembali file dan pastikan prediksinya sama dengan instance sebelum serialisasi.
Keberhasilan `joblib.load` tanpa perbandingan prediksi belum cukup sebagai verifikasi.

## Evaluasi dan pelaporan

- Hasilkan prediksi aktual pada seluruh test, termasuk vektor kosong. Laporkan
  jumlah vektor kosong; jangan menghapusnya untuk memperbaiki metrik.
- Prioritaskan **Macro F1**. Sertakan Accuracy, Precision/Recall dengan averaging
  eksplisit, Weighted F1, dan Precision/Recall/F1/support setiap kelas. Dokumentasikan
  penanganan pembagi nol.
- Confusion matrix harus menyebut urutan kelas: baris aktual, kolom prediksi.
  Simpan counts; normalisasi per baris boleh ditambahkan untuk melihat recall kelas.
- Simpan prediksi per baris berikut identitas dan label aktual, classification
  report, tabel model, serta grafik dari angka eksperimen aktual. Sertakan baseline
  kelas mayoritas yang dipilih dari training dan analisis salah klasifikasi.
- Verifikasi model yang dimuat ulang: prediksi sesuai file prediksi, metrik dapat
  dihitung ulang, dan total confusion matrix sama dengan jumlah test. Jangan
  menyimpulkan validasi hanya dari pesan “berhasil”.
- Beri peringkat menurut Macro F1 sesuai rencana. Satu test tidak membuktikan
  signifikansi statistik. Jelaskan kelemahan Neutral, proksi rating, cleaning,
  provenance, dan deduplikasi. Jangan mengklaim penyebab sarkasme/bahasa tanpa bukti.
- Laporkan perintah, hasil tes, lokasi artifact, dan pekerjaan yang belum selesai.
  Jangan membuat angka, grafik, klaim eksekusi, atau anotasi manusia fiktif.

## Streamlit bila sudah masuk scope

Baca skill lokal `developing-with-streamlit` sebelum menulis kode Streamlit dan
gunakan dokumentasi yang sesuai versi terpasang. Dashboard memuat model/vectorizer
tersimpan tanpa fit saat startup/rerun. Gunakan cleaning sesuai manifest, urutan
fitur/kelas yang benar, serta penanganan input kosong atau vektor nol.

LinearSVC tidak menyediakan `predict_proba`; jangan menampilkan decision score
sebagai probabilitas. Kalibrasi, jika diminta, harus memakai training/validasi.
Pisahkan overview/EDA, perbandingan, evaluasi, dan prediksi sesuai PRD. Ambil angka
dari artifact evaluasi terverifikasi, bukan metrik hardcoded. Uji konsistensi hasil
antara notebook, modul, model yang dimuat ulang, dan tampilan dashboard.
