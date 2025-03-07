import streamlit as st
import nltk
import json
import random
import ssl

# NLTK-Ressourcen herunterladen (falls noch nicht vorhanden)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stemmer/porter')
except LookupError:
    nltk.download('porter')

def load_intents(file_path='intents.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        intents = json.load(file)
    return intents

def process_input(user_input, intents):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stemmer = nltk.stem.PorterStemmer()

    words = nltk.word_tokenize(user_input)
    words = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in words]

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_words = nltk.word_tokenize(pattern)
            pattern_words = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in pattern_words]

            if all(word in words for word in pattern_words):
                return random.choice(intent['responses'])

    return "Ich verstehe das nicht."

def main():
    st.title("Chatbot")
    st.write("Hallo! Wie kann ich dir helfen?")

    intents = load_intents()

    user_input = st.text_input("Du:", "")

    if user_input:
        response = process_input(user_input, intents)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
