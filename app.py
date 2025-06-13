import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load model and tokenizer once
@st.cache_resource
def load_resources():
    model = load_model("next_word_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()
max_len = 5  # Replace with your actual training-time max_len

# Reverse word index for easy lookup
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# UI
st.title("Next Word Prediction ")
st.markdown("""
This app works like a smart word recommender — just like how your phone suggests the next word while typing.
Based on your input sentence, it uses a trained neural network to predict and display the **top 3 possible next words**.

It’s useful for building **autocomplete systems, AI writing assistants, or text generation tools**.
""")

input_text = st.text_input("Type a sentence:")

if input_text.strip():
    try:
        token_text = tokenizer.texts_to_sequences([input_text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=max_len, padding='pre')
        prediction = model.predict(padded_token_text, verbose=0)[0]

        # Get top-3 predicted indices
        top_3_indices = prediction.argsort()[-3:][::-1]

        top_3_words = []
        for idx in top_3_indices:
            word = index_to_word.get(idx)
            if word:
                top_3_words.append(word)

        if top_3_words:
            st.markdown("### 🔮 Top 3 Suggestions:")
            for i, word in enumerate(top_3_words, 1):
                full_sentence = input_text.strip() + " " + word
                st.write(f"**{i}.** {full_sentence}")
        else:
            st.warning("Couldn't find valid next words.")

    except Exception as e:
        st.error(f"Error: {e}")
