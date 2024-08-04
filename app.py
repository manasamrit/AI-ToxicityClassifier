import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the Tfidf and model once
@st.cache_resource
def load_tfidf() -> TfidfVectorizer:
    with open("tf_idf.pkt", "rb") as file:
        tfidf = pickle.load(file)
    return tfidf

@st.cache_resource
def load_model() -> MultinomialNB:
    with open("toxicity_model.pkt", "rb") as file:
        nb_model = pickle.load(file)
    return nb_model

def toxicity_prediction(text: str) -> str:
    tfidf = load_tfidf()
    nb_model = load_model()
    text_tfidf = tfidf.transform([text])
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name

# Streamlit app
st.header("Toxicity Detection App")
st.subheader("Input your text")

text_input = st.text_input("Enter your text")

if text_input:
    if st.button("Analyze"):
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        st.info(f"The result is {result}.")
