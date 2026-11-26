import streamlit as st

from dashboard.components import count_chart, metric_row, page_header
from dashboard.data import get_bundle, get_reviews, signature
from src.modeling import CLASSES, MODEL_NAMES

page_header('Overview', 'Apa cerita di balik ulasan GTA VI?',
            'Eksplorasi ulasan pengguna dan bandingkan tiga model untuk memprediksi '
            '**Negative**, **Neutral**, atau **Positive** dari teks.')
token = signature()
bundle = get_bundle(token)
reviews = get_reviews(token)
result = bundle.evaluation
metric_row([
    ('Ulasan dianalisis', f'{len(reviews):,}'),
    ('Data training', f'{sum(bundle.training_manifest["training_class_counts"].values()):,}'),
    ('Data testing', f'{result["test_rows"]:,}'),
    ('Kelas sentimen', len(CLASSES)),
])

with st.container(border=True):
    st.subheader('Distribusi sentimen')
    st.caption('Seluruh dataset processed; label dibentuk dari rating pengguna.')
    count_chart(reviews, 'sentiment', 'Jumlah ulasan Negative, Neutral, dan Positive pada dataset analisis', CLASSES)

with st.container(horizontal=True, gap='medium'):
    with st.container(border=True, width=440):
        st.subheader('Eksperimen yang dapat dibandingkan')
        st.markdown('Multinomial Naive Bayes, Logistic Regression, dan Linear SVM memakai '
                    'fitur TF-IDF serta pembagian data yang sama. Teks bersih identik '
                    'dikelompokkan agar tidak melintasi training dan testing.')
        st.caption('Data ulasan: ' + str(reviews.published_date.min()) + ' sampai ' + str(reviews.published_date.max()))
    with st.container(border=True, width=440):
        best = result['ranking_macro_f1'][0]
        st.subheader('Terbaik pada Macro F1')
        st.markdown(f'### {MODEL_NAMES[best]}')
        st.metric('Macro F1 pada test', f'{result["models"][best]["metrics"]["f1_macro"]:.2%}')
        st.caption('Peringkat dari satu test set; bukan bukti keunggulan pada semua data.')

st.subheader('Cara membaca label')
st.markdown('**Negative: rating 0–4** · **Neutral: rating 5–7** · **Positive: rating 8–10**. '
            'Model mempelajari hubungan teks dengan kategori rating tersebut, bukan penilaian sentimen manusia.')
st.caption('Sumber yang diberikan pemilik proyek: [Kaggle — GTA VI user reviews]'
           '(https://www.kaggle.com/datasets/ektarr/gta-vi-user-reviews/data). '
           'Versi unduhan, lisensi, dan arti review_id belum terverifikasi independen. '
           'Dataset memakai kebijakan deduplikasi dan cleaning legacy yang didokumentasikan dalam README.')
