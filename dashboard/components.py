"""Shared display helpers and analytical charts; no model training."""

import altair as alt
import pandas as pd
import streamlit as st

from src.modeling import CLASSES, MODEL_NAMES

METRICS = {'accuracy': 'Accuracy', 'precision_macro': 'Precision macro',
           'recall_macro': 'Recall macro', 'f1_macro': 'Macro F1', 'f1_weighted': 'Weighted F1'}


def page_header(section, title, description):
    st.caption(f'GTA VI SENTIMENT LAB  /  {section.upper()}')
    st.title(title)
    st.markdown(description)


def metric_row(items):
    with st.container(horizontal=True, gap='medium'):
        for label, value in items:
            st.metric(label, value, border=True, width=210)


def count_chart(frame, column, title, order=None):
    counts = frame[column].value_counts().rename_axis('Kategori').reset_index(name='Ulasan')
    counts['Proporsi'] = counts.Ulasan / len(frame)
    chart = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('Kategori:N', title=None, sort=order or '-y', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Ulasan:Q', scale=alt.Scale(zero=True), title='Jumlah ulasan'),
        color=alt.Color('Kategori:N', sort=order, legend=None),
        tooltip=['Kategori:N', alt.Tooltip('Ulasan:Q', format=','), alt.Tooltip('Proporsi:Q', format='.1%')],
    ).properties(height=270)
    st.altair_chart(chart, width='stretch', alt=title)


def comparison_frame(result):
    rows = [{'Model': MODEL_NAMES[name], **{label: info['metrics'][key] for key, label in METRICS.items()},
             'Pilihan Macro F1': name == result['ranking_macro_f1'][0]}
            for name, info in result['models'].items()]
    return pd.DataFrame(rows).sort_values('Macro F1', ascending=False).reset_index(drop=True)


def score_table(frame, alt_text):
    display = frame.copy()
    configuration = {}
    for name in frame.columns:
        if name in {*METRICS.values(), 'Precision', 'Recall', 'F1'}:
            display[name] = frame[name] * 100
            configuration[name] = st.column_config.NumberColumn(name, format='%.2f%%')
    st.dataframe(display, hide_index=True, width='stretch', column_config=configuration, alt=alt_text)


def confusion_chart(matrix, normalized=False):
    cells = []
    for i, actual in enumerate(CLASSES):
        support = sum(matrix[i])
        for j, predicted in enumerate(CLASSES):
            count = matrix[i][j]
            cells.append({'Aktual': actual, 'Prediksi': predicted, 'Jumlah': count,
                          'Nilai': count / support if normalized and support else count})
    source = pd.DataFrame(cells)
    base = alt.Chart(source).encode(
        x=alt.X('Prediksi:N', sort=CLASSES, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Aktual:N', sort=CLASSES),
        tooltip=['Aktual:N', 'Prediksi:N', 'Jumlah:Q', alt.Tooltip('Nilai:Q', format='.1%' if normalized else ',')])
    heatmap = base.mark_rect(cornerRadius=5).encode(
        color=alt.Color('Nilai:Q', scale=alt.Scale(scheme='blues', domain=[0, 1] if normalized else [0, source.Nilai.max()]), legend=None))
    threshold = .5 if normalized else source.Nilai.max() / 2
    labels = base.mark_text(fontSize=17, fontWeight='bold').encode(
        text=alt.Text('Nilai:Q', format='.1%' if normalized else ','),
        color=alt.condition(alt.datum.Nilai > threshold, alt.value('white'), alt.value('#10243b')))
    st.altair_chart((heatmap + labels).properties(height=330), width='stretch',
                   alt='Confusion matrix: baris kelas aktual, kolom prediksi; seluruh ulasan test dihitung')
