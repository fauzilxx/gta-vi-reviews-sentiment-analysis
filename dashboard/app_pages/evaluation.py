import pandas as pd
import streamlit as st

from dashboard.components import confusion_chart, metric_row, page_header, score_table
from dashboard.data import get_bundle, signature
from src.modeling import CLASSES, MODEL_NAMES

page_header('Model evaluation', 'Lihat kelas yang berhasil dan yang terlewat',
            'Confusion matrix dan classification report dihitung dari prediksi aktual. '
            'Semua ulasan test, termasuk vektor kosong, tetap masuk evaluasi.')
result = get_bundle(signature()).evaluation
name = st.segmented_control('Algoritma evaluasi', list(MODEL_NAMES),
                            format_func=MODEL_NAMES.__getitem__, default=result['ranking_macro_f1'][0],
                            required=True, key='evaluation_model')
info = result['models'][name or result['ranking_macro_f1'][0]]
metric_row([('Macro F1', f'{info["metrics"]["f1_macro"]:.2%}'),
            ('Accuracy', f'{info["metrics"]["accuracy"]:.2%}'),
            ('Recall Neutral', f'{info["classification_report"]["Neutral"]["recall"]:.2%}'),
            ('Salah klasifikasi', f'{info["error_count"]:,}')])
with st.container(border=True):
    st.subheader('Confusion matrix')
    normalized = st.toggle('Normalisasi per kelas aktual', value=False, key='normalized_matrix')
    confusion_chart(info['confusion_matrix'], normalized)
    st.caption('Baris = label aktual; kolom = prediksi model. '
               + ('Setiap baris berjumlah 100%; tooltip tetap menampilkan jumlah ulasan.' if normalized
                  else 'Angka menunjukkan jumlah ulasan; diagonal menunjukkan prediksi benar.'))

st.subheader('Classification report')
rows = []
for label in CLASSES + ['macro avg', 'weighted avg']:
    report = info['classification_report'][label]
    rows.append({'Kelas / rata-rata': label, 'Precision': report['precision'], 'Recall': report['recall'],
                 'F1': report['f1-score'], 'Support': int(report['support'])})
score_table(pd.DataFrame(rows), 'Precision, Recall, F1, dan support setiap kelas serta macro dan weighted average')
st.caption('Precision, Recall, dan F1 dalam persen; Support adalah jumlah ulasan aktual. '
           'Macro average memberi bobot sama per kelas; weighted average mengikuti support.')
neutral = info['classification_report']['Neutral']
st.markdown(f'**Neutral:** {info["confusion_matrix"][1][1]:,} dari {int(neutral["support"]):,} ulasan dikenali dengan benar. '
            f'Precision {neutral["precision"]:.2%} dan recall {neutral["recall"]:.2%}.')
st.caption(f'{result["zero_vector_test_rows"]} ulasan test tidak memiliki fitur yang dikenali vectorizer. '
           'Ulasan tersebut tetap dihitung dalam semua metrik di atas.')
