import altair as alt
import streamlit as st

from dashboard.components import count_chart, metric_row, page_header
from dashboard.data import get_eda, get_reviews, signature
from src.modeling import CLASSES

page_header('Exploratory data analysis', 'Kenali data sebelum membaca skor',
            'Semua grafik di halaman ini berasal dari dataset processed yang sama dengan eksperimen. '
            'Arahkan kursor ke grafik untuk melihat nilainya.')
token = signature()
reviews = get_reviews(token)

with st.container(border=True):
    st.subheader('Distribusi sentimen')
    count_chart(reviews, 'sentiment', 'Distribusi label sentimen dataset processed', CLASSES)
    neutral_fraction = reviews.sentiment.eq('Neutral').mean()
    st.caption(f'Kelas Neutral mencakup {neutral_fraction:.2%} data. Macro F1 memberi bobot yang sama kepada setiap kelas.')
with st.container(border=True):
    st.subheader('Distribusi rating')
    ratings = reviews.groupby('score').size().reindex(range(11), fill_value=0).rename('Ulasan').reset_index()
    chart = alt.Chart(ratings).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X('score:O', title='Rating pengguna (0–10)', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Ulasan:Q', scale=alt.Scale(zero=True), title='Jumlah ulasan'),
        tooltip=[alt.Tooltip('score:O', title='Rating'), 'Ulasan:Q']).properties(height=260)
    st.altair_chart(chart, width='stretch', alt='Frekuensi rating pengguna pada skala 0 sampai 10')
with st.container(border=True):
    st.subheader('Distribusi platform')
    count_chart(reviews, 'platform', 'Jumlah ulasan per platform pada dataset processed')

eda = get_eda(token)
st.subheader('Panjang ulasan')
metric_row([('Median karakter', f'{reviews.text_length.median():,.0f}'),
            ('Rata-rata karakter', f'{reviews.text_length.mean():,.0f}'),
            ('Terpanjang', f'{reviews.text_length.max():,}')])
histogram = alt.Chart(eda['length_histogram']).mark_bar().encode(
    x=alt.X('Awal:Q', bin='binned', title='Panjang teks bersih (karakter)', scale=alt.Scale(zero=True)),
    x2='Akhir:Q', y=alt.Y('Ulasan:Q', title='Jumlah ulasan', scale=alt.Scale(zero=True)),
    tooltip=[alt.Tooltip('Awal:Q', format='.0f'), alt.Tooltip('Akhir:Q', format='.0f'), 'Ulasan:Q'],
).properties(height=290)
st.altair_chart(histogram, width='stretch', alt='Histogram panjang seluruh teks bersih tanpa pemotongan outlier')
st.caption('Seluruh rentang panjang ditampilkan; panjang dihitung setelah cleaning legacy-v1.')

st.subheader('Kata yang sering muncul')
limit = st.segmented_control('Jumlah kata', [10, 20, 30], default=20, required=True, key='word_limit')
words = eda['words'].head(limit or 20)
word_chart = alt.Chart(words).mark_bar(cornerRadiusEnd=4).encode(
    y=alt.Y('Kata:N', sort='-x', title=None),
    x=alt.X('Kemunculan:Q', title='Jumlah kemunculan token', scale=alt.Scale(zero=True)),
    tooltip=['Kata:N', 'Kemunculan:Q']).properties(height=max(300, len(words) * 22))
st.altair_chart(word_chart, width='stretch', alt='Kata paling sering muncul dalam seluruh teks bersih dataset')
st.caption('Frekuensi kemunculan token, bukan bobot TF-IDF atau pengaruh kata pada prediksi. '
           'Stopword tidak dihapus karena teks multibahasa. Statistik deskriptif ini tidak melatih atau mengubah vocabulary model.')
