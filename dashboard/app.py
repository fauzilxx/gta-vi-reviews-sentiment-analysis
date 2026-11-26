"""Local Streamlit dashboard entrypoint: streamlit run dashboard/app.py."""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from dashboard.data import get_bundle, signature

st.set_page_config(page_title='GTA VI · Sentiment lab', page_icon=':material/analytics:', layout='wide')

with st.sidebar:
    st.title('GTA VI')
    st.caption('SENTIMENT LAB')
    st.markdown('Dari ulasan pengguna ke evaluasi tiga model klasifikasi.')

pages = st.navigation([
    st.Page('app_pages/overview.py', title='Overview', icon=':material/dashboard:', default=True),
    st.Page('app_pages/eda.py', title='Exploratory data analysis', icon=':material/query_stats:', url_path='eda'),
    st.Page('app_pages/comparison.py', title='Model comparison', icon=':material/leaderboard:', url_path='comparison'),
    st.Page('app_pages/evaluation.py', title='Model evaluation', icon=':material/fact_check:', url_path='evaluation'),
    st.Page('app_pages/prediction.py', title='Sentiment prediction', icon=':material/chat:', url_path='prediction'),
])

try:
    current_signature = signature()
    get_bundle(current_signature)
except Exception:
    logging.getLogger(__name__).exception('Gagal memuat artifact dashboard')
    st.error('File eksperimen tidak tersedia, berubah, atau versi paket tidak sesuai. '
             'Periksa artifact dan environment menggunakan petunjuk README.', title='Eksperimen belum dapat dimuat')
    st.stop()

with st.sidebar:
    st.caption('Label berasal dari rating 0–10, bukan anotasi sentimen manusia.')
    st.caption('Eksperimen grouped_v1 · TF-IDF · 3 kelas')

pages.run()
