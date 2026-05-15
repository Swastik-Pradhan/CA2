import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("imdb_sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

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

        # Transform input using TF-IDF
        review_vector = tfidf.transform([review])

        # Predict
        prediction = model.predict(review_vector)

        # Output
        if prediction[0] == 1:
            st.success("😊 Positive Review")

        else:
            st.error("😞 Negative Review")
