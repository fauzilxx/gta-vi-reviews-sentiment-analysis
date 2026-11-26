import logging
import streamlit as st

from dashboard.components import page_header
from dashboard.data import get_bundle, signature
from src.modeling import MODEL_NAMES
from src.predict import predict_sentiment

page_header('Sentiment prediction', 'Coba satu ulasan baru',
            'Masukkan ulasan dan pilih algoritma. Model akan memprediksi '
            'kategori **Negative**, **Neutral**, atau **Positive** dari teks.')
bundle = get_bundle(signature())
with st.form('prediction_form', border=True):
    selected = st.segmented_control('Algoritma prediksi', list(MODEL_NAMES),
                                    format_func=MODEL_NAMES.__getitem__, default=bundle.evaluation['ranking_macro_f1'][0],
                                    required=True, key='prediction_model')
    review = st.text_area('Teks ulasan', height=180, key='review_text',
                          placeholder='Contoh: The story is engaging, but the performance needs improvement.')
    submitted = st.form_submit_button('Prediksi sentimen', type='primary', icon=':material/analytics:',
                                     width='stretch', key='predict_submit')

st.caption('Label training dibentuk dari rating: Negative 0–4, Neutral 5–7, Positive 8–10. '
           'Hasil merupakan prediksi kategori rating dari teks, bukan penilaian manusia atau kepastian sentimen.')

if submitted:
    try:
        prediction = predict_sentiment(review, selected, bundle)
    except ValueError as error:
        st.warning(str(error), title='Ulasan belum dapat diprediksi')
    except Exception:
        logging.getLogger(__name__).exception('Kesalahan saat prediksi')
        st.error('Prediksi gagal diproses. Periksa model dan coba kembali.', title='Prediksi belum tersedia')
    else:
        with st.container(border=True):
            st.caption('HASIL PREDIKSI')
            st.subheader(prediction['label'])
            st.markdown(f'Model: **{MODEL_NAMES[prediction["model"]]}**')
            st.caption('Model mengenali fitur teks dari ulasan ini. Hasil dapat keliru, terutama untuk '
                       'ulasan Neutral, sarkasme, atau bahasa yang kurang terwakili dalam data.')

st.markdown('Bahasa dan pola kata di luar vocabulary training dapat tidak dikenali. '
            'Tidak ada confidence atau probabilitas yang ditampilkan karena keluaran ketiga model tidak setara.')
