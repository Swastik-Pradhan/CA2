import streamlit as st
import pickle
import os

# Get current folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer
model_path = os.path.join(BASE_DIR, "imdb_sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
tfidf = pickle.load(open(vectorizer_path, "rb"))

# Page config
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Title
st.title("🎬 IMDB Sentiment Analysis")

st.write("Predict whether a movie review is Positive or Negative.")

# Input box
review = st.text_area(
    "Enter Movie Review",
    placeholder="Example: This movie was fantastic and very entertaining."
)

# Prediction
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        review_vector = tfidf.transform([review])

        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")
