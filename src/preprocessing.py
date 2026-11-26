"""
src/preprocessing.py
Pipeline Preprocessing dan Pelabelan Sentimen untuk Dataset Ulasan GTA VI.
Sesuai panduan PRD Phase 2:
- Memeriksa dan menangani duplikasi review_id terlebih dahulu
- Memeriksa duplikasi teks secara terpisah
- Menjaga ulasan multibahasa tanpa stopword removal / stemming buta
- Menjaga kata negasi ('not', 'no', 'never', dll.)
- Membuat label sentimen 3 kelas (0-4: Negative, 5-7: Neutral, 8-10: Positive)
- Menyimpan hasil ke data/processed/processed_reviews.csv
"""

import os
import re
import html
import unicodedata
import pandas as pd
import numpy as np

# Pola kontraksi umum bahasa Inggris untuk mempertahankan kata negasi
CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bcan’t\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bwon’t\b": "will not",
    r"\bdon't\b": "do not",
    r"\bdon’t\b": "do not",
    r"\bdidn't\b": "did not",
    r"\bdidn’t\b": "did not",
    r"\bdoesn't\b": "does not",
    r"\bdoesn’t\b": "does not",
    r"\bisn't\b": "is not",
    r"\bisn’t\b": "is not",
    r"\baren't\b": "are not",
    r"\baren’t\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bwasn’t\b": "was not",
    r"\bweren't\b": "were not",
    r"\bweren’t\b": "were not",
    r"\bhaven't\b": "have not",
    r"\bhaven’t\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bhasn’t\b": "has not",
    r"\bhadn't\b": "had not",
    r"\bhadn’t\b": "had not",
    r"\bwouldn't\b": "would not",
    r"\bwouldn’t\b": "would not",
    r"\bshouldn't\b": "should not",
    r"\bshouldn’t\b": "should not",
    r"\bcouldn't\b": "could not",
    r"\bcouldn’t\b": "could not",
    r"\bit's\b": "it is",
    r"\bit’s\b": "it is",
    r"\bi'm\b": "i am",
    r"\bi’m\b": "i am",
    r"\bthat's\b": "that is",
    r"\bthat’s\b": "that is",
    r"\byou're\b": "you are",
    r"\byou’re\b": "you are",
    r"\bthey're\b": "they are",
    r"\bthey’re\b": "they are",
    r"\bwe're\b": "we are",
    r"\bwe’re\b": "we are"
}

def clean_text(text: str, *, version: str = "legacy-v1") -> str:
    """
    Membersihkan teks ulasan dengan:
    1. Unescape HTML entities (&quot;, &amp;, dll.)
    2. Menghapus tag HTML dan URL
    3. Ekspansi kontraksi bahasa Inggris agar negasi (not/no) terjaga
    4. Case folding ke lowercase
    5. Normalisasi spasi dan tanda baca berulang
    6. Mempertahankan karakter huruf/angka Unicode dari berbagai bahasa
    Tanpa stopword removal atau stemming buta agar struktur sentimen multibahasa tidak rusak.
    """
    # Artifact lama memakai legacy-v1. Jangan mengganti preprocessing inferensi
    # tanpa membangun dataset/vectorizer versi baru dengan konfigurasi yang sama.
    if version not in {"legacy-v1", "unicode-v2"}:
        raise ValueError(f"Versi cleaning tidak dikenal: {version}")
    if not isinstance(text, str):
        return ""
    if version == "unicode-v2":
        text = unicodedata.normalize("NFC", text)
    
    # 1. Unescape HTML entities
    text = html.unescape(text)
    
    # 2. Hapus URL web
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 3. Hapus tag HTML
    text = re.sub(r'<.*?>', ' ', text)
    
    # 4. Ekspansi kontraksi bahasa Inggris
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    # 5. Case folding
    text = text.lower()
    
    # 6. Ganti karakter newline/tab/carriage return dengan spasi
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # 7. Hapus karakter non-alphanumeric (tanda baca, simbol), pertahankan Unicode letters/digits
    # Regex [^\w\s] mempertahankan huruf beraksen (á, é, ñ, ç) serta alfabet lain (Cyrillic, dll.)
    if version == "legacy-v1":
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    else:
        text = ''.join(
            char if (char.isalnum() or char == '_' or char.isspace()
                     or unicodedata.category(char).startswith('M')) else ' '
            for char in text
        )
    
    # 8. Hapus angka yang berdiri sendiri (opsional: biarkan angka jika bagian dari kata seperti gta6)
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # 9. Normalisasi multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def assign_sentiment(score: int) -> str:
    """
    Memberikan label sentimen 3 kelas berdasarkan skor rating:
    - 0 s.d. 4 : Negative
    - 5 s.d. 7 : Neutral
    - 8 s.d. 10 : Positive
    """
    if isinstance(score, (bool, np.bool_)):
        raise ValueError("Rating harus berupa bilangan bulat 0–10, bukan boolean")
    try:
        numeric_score = float(score)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Rating harus berupa bilangan bulat 0–10") from exc
    if not np.isfinite(numeric_score) or not numeric_score.is_integer() or not 0 <= numeric_score <= 10:
        raise ValueError("Rating harus berupa bilangan bulat 0–10")
    score = int(numeric_score)
    if score <= 4:
        return 'Negative'
    elif score <= 7:
        return 'Neutral'
    else:
        return 'Positive'

def run_preprocessing(
    raw_path: str = "data/raw/gta_vi_reviews.csv",
    output_path: str = "data/processed/processed_reviews.csv",
    cleaning_version: str = "legacy-v1"
):
    if os.path.exists(output_path):
        raise FileExistsError(f"Artifact sudah ada; gunakan output_path versi baru: {output_path}")
    if cleaning_version not in {"legacy-v1", "unicode-v2"}:
        raise ValueError(f"Versi cleaning tidak dikenal: {cleaning_version}")
    print("=" * 60)
    print("GTA VI REVIEWS - PREPROCESSING PIPELINE (PHASE 2)")
    print("=" * 60)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File raw dataset tidak ditemukan di: {raw_path}")
        
    print(f"\n[1] Membaca raw dataset dari: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    required_columns = {'review_id', 'score', 'review_text', 'platform', 'published_date'}
    if not required_columns.issubset(df_raw.columns):
        raise ValueError(f"Kolom wajib tidak lengkap: {required_columns - set(df_raw.columns)}")
    initial_count = len(df_raw)
    print(f"    Total data awal: {initial_count:,} baris")
    print(f"    Kolom awal: {list(df_raw.columns)}")
    
    # Cek Missing Value Awal
    missing_initial = df_raw.isnull().sum()
    print("\n[2] Pemeriksaan Missing Values Awal:")
    for col, count in missing_initial.items():
        print(f"    - {col}: {count} missing")
        
    df = df_raw.copy()
    
    # Drop missing values jika ada
    df = df.dropna(subset=['review_id', 'score', 'review_text'])
    # Validasi sebelum astype(int), agar rating pecahan tidak terpotong diam-diam.
    df['score'].map(assign_sentiment)
    after_na = len(df)
    dropped_na = initial_count - after_na
    print(f"    Data setelah penanganan missing value: {after_na:,} baris (Dihapus: {dropped_na})")
    
    # [3] Penanganan Duplikasi review_id terlebih dahulu
    print("\n[3] Pemeriksaan Duplikasi 'review_id':")
    dup_id_count = df.duplicated(subset=['review_id']).sum()
    print(f"    Jumlah review_id duplikat: {dup_id_count:,} baris")
    df = df.drop_duplicates(subset=['review_id'], keep='first')
    after_dup_id = len(df)
    print(f"    Data setelah penghapusan duplikasi review_id: {after_dup_id:,} baris (Dihapus: {dup_id_count})")
    
    # [4] Pemeriksaan Duplikasi review_text secara terpisah
    print("\n[4] Pemeriksaan Duplikasi 'review_text' (Terpisah):")
    dup_text_count = df.duplicated(subset=['review_text']).sum()
    print(f"    Jumlah review_text duplikat: {dup_text_count:,} baris")
    df = df.drop_duplicates(subset=['review_text'], keep='first')
    after_dup_text = len(df)
    print(f"    Data setelah penghapusan duplikasi review_text: {after_dup_text:,} baris (Dihapus: {dup_text_count})")
    
    # [5] Pembersihan Teks (clean_text)
    print("\n[5] Melakukan Pembersihan Teks (clean_text)...")
    df['clean_text'] = df['review_text'].map(lambda text: clean_text(text, version=cleaning_version))
    
    # Filter teks hasil pembersihan yang kosong atau terlalu pendek (< 3 karakter / < 2 kata)
    words_count = df['clean_text'].apply(lambda t: len(t.split()))
    short_texts = (words_count < 2).sum()
    print(f"    Jumlah teks hasil pembersihan yang terlalu pendek (< 2 kata): {short_texts:,}")
    df = df[words_count >= 2].copy()
    after_clean_filter = len(df)
    print(f"    Data setelah pemfilteran teks bermakna: {after_clean_filter:,} baris (Dihapus: {short_texts})")
    
    # [6] Pembentukan Label Sentimen 3 Kelas
    print("\n[6] Pelabelan Sentimen 3 Kelas (Berdasarkan rating disepakati):")
    df['sentiment'] = df['score'].apply(assign_sentiment)
    df['score'] = df['score'].astype(int)
    
    sentiment_dist = df['sentiment'].value_counts()
    sentiment_pct = df['sentiment'].value_counts(normalize=True) * 100
    for label in ['Negative', 'Neutral', 'Positive']:
        cnt = sentiment_dist.get(label, 0)
        pct = sentiment_pct.get(label, 0.0)
        print(f"    - {label:<8}: {cnt:>6,} ulasan ({pct:5.2f}%)")
        
    # [7] Simpan Dataset Bersih
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n[7] Dataset hasil preprocessing berhasil disimpan ke:")
    print(f"    -> {output_path}")
    print(f"    Total baris akhir: {len(df):,} baris")
    print(f"    Kolom akhir: {list(df.columns)}")
    
    # Ringkasan Pengurangan Data
    total_removed = initial_count - len(df)
    pct_removed = (total_removed / initial_count) * 100
    print("\n" + "=" * 60)
    print("RINGKASAN TAHAPAN PEMBERSIHAN DATA:")
    print(f"  * Total Data Mentah (Raw)      : {initial_count:>6,} (100.00%)")
    print(f"  * Dihapus (Missing Value)       : {dropped_na:>6,}")
    print(f"  * Dihapus (Duplikasi ID)        : {dup_id_count:>6,}")
    print(f"  * Dihapus (Duplikasi Teks)      : {dup_text_count:>6,}")
    print(f"  * Dihapus (Teks Kosong/Pendek)  : {short_texts:>6,}")
    print(f"  * Total Data Dihapus            : {total_removed:>6,} ({pct_removed:.2f}%)")
    print(f"  * Total Data Bersih (Final)     : {len(df):>6,} ({100 - pct_removed:.2f}%)")
    print("=" * 60)
    
    return {
        'initial_count': initial_count,
        'final_count': len(df),
        'dropped_na': dropped_na,
        'dropped_dup_id': dup_id_count,
        'dropped_dup_text': dup_text_count,
        'dropped_short_text': short_texts,
        'total_removed': total_removed,
        'sentiment_distribution': sentiment_dist.to_dict()
    }

if __name__ == "__main__":
    run_preprocessing()
