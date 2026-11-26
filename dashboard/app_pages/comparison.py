import altair as alt
import streamlit as st

from dashboard.components import METRICS, comparison_frame, metric_row, page_header, score_table
from dashboard.data import get_bundle, signature
from src.modeling import MODEL_NAMES

page_header('Model comparison', 'Tiga model. Satu pengujian yang sama.',
            'Bandingkan performa dari prediksi aktual pada test set yang sama. '
            '**Macro F1** menjadi acuan utama karena setiap kelas mendapat bobot setara.')
result = get_bundle(signature()).evaluation
best = result['ranking_macro_f1'][0]
best_metrics = result['models'][best]['metrics']
st.badge(f'Terbaik pada Macro F1: {MODEL_NAMES[best]}', icon=':material/workspace_premium:', color='green')
metric_row([('Macro F1 terbaik', f'{best_metrics["f1_macro"]:.2%}'),
            ('Ulasan test', f'{result["test_rows"]:,}'),
            ('Macro F1 baseline mayoritas', f'{result["majority_baseline"]["metrics"]["f1_macro"]:.2%}')])
frame = comparison_frame(result)
score_table(frame, 'Perbandingan Accuracy, Precision macro, Recall macro, Macro F1, dan Weighted F1 ketiga model')
st.caption('Semua skor dalam persen. Precision dan Recall memakai macro average; pembagi nol diberi nilai 0.')

long = frame.melt(id_vars='Model', value_vars=list(METRICS.values()), var_name='Metrik', value_name='Skor')
chart = alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
    x=alt.X('Metrik:N', sort=list(METRICS.values()), axis=alt.Axis(labelAngle=0), title=None),
    xOffset='Model:N', y=alt.Y('Skor:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%'), title='Skor'),
    color=alt.Color('Model:N', legend=alt.Legend(orient='bottom')),
    tooltip=['Model:N', 'Metrik:N', alt.Tooltip('Skor:Q', format='.2%')]).properties(height=340)
st.altair_chart(chart, width='stretch', alt='Perbandingan lima metrik ketiga model, seluruh sumbu skor dari nol sampai seratus persen')

with st.container(border=True):
    st.subheader('Mengapa tidak memilih dari Accuracy saja?')
    st.markdown('Accuracy dapat terlihat tinggi saat kelas mayoritas dikenali dengan baik, '
                'meskipun Neutral sering salah. Lihat performa setiap kelas pada halaman Model evaluation.')
    st.caption('LR dan SVM memakai class_weight=balanced; Naive Bayes memakai prior empiris training. '
               'Perbandingan mencakup konfigurasi tersebut dan bukan isolasi pengaruh algoritma saja. '
               'Satu held-out test tidak membuktikan signifikansi statistik; test tidak digunakan untuk tuning.')
