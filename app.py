import streamlit as st
import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(page_title="FAQ Chatbot")

st.title("🤖 FAQ Chatbot")
st.write("Ask a question from the FAQ database.")

# Load Dataset
faq = pd.read_csv("faq_data.csv")

# Clean Text
def clean_text(text):
    text = str(text).lower()

    for p in string.punctuation:
        text = text.replace(p, "")

    return text

faq["Question"] = faq["Question"].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    faq["Question"]
)

# Chatbot Function
def get_answer(user_question):

    user_question = clean_text(
        user_question
    )

    user_vector = vectorizer.transform(
        [user_question]
    )

    similarity = cosine_similarity(
        user_vector,
        faq_vectors
    )

    score = similarity.max()

    if score < 0.3:
        return "Sorry, I couldn't find a matching answer."

    index = similarity.argmax()

    return faq.iloc[index]["Answer"]

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input(
    "Enter your question:"
)

if st.button("Get Answer"):

    if user_input:

        answer = get_answer(
            user_input
        )

        st.session_state.messages.append(
            ("You", user_input)
        )

        st.session_state.messages.append(
            ("Bot", answer)
        )

for sender, message in st.session_state.messages:

    st.write(
        f"**{sender}:** {message}"
    )
