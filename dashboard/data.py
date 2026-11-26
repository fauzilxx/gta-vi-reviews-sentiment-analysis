"""Streamlit cache wrappers; processing and inference live in src.predict."""

import streamlit as st

from src.predict import PROJECT_ROOT, artifact_signature, eda_summary, load_model_bundle, load_reviews


@st.cache_resource(max_entries=1, show_spinner='Memeriksa dan memuat model eksperimen…')
def get_bundle(signature):
    return load_model_bundle(PROJECT_ROOT)


@st.cache_data(max_entries=1, show_spinner='Memuat dataset analisis…')
def get_reviews(signature):
    return load_reviews(PROJECT_ROOT)


@st.cache_data(max_entries=1, show_spinner='Menghitung ringkasan kata dan panjang ulasan…')
def get_eda(signature):
    return eda_summary(get_reviews(signature), get_bundle(signature).vectorizer)


def signature():
    return artifact_signature(PROJECT_ROOT)
