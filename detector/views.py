import re
import numpy as np
import spacy
import tensorflow as tf
from django.shortcuts import render
from tensorflow.keras.preprocessing.sequence import pad_sequences

# spaCy Model & Trained Keras Model 
nlp = spacy.load('en_core_web_sm')
model = tf.keras.models.load_model('hate_speech_lstm.h5')

# Preprocessing helper function 
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'[\s]+', ' ', text)
    
    # Lemmatization & Stopwords removal
    doc = nlp(text)
    clean_tokens = [word.lemma_ for word in doc if not word.is_stop]
    return ' '.join(clean_tokens)

def predict_tweet(request):
    result = None
    confidence = None
    user_input = ""

    if request.method == "POST":
        user_input = request.POST.get('tweet_text', '')
        if user_input:
            # 1. Clean Text
            cleaned = preprocess_text(user_input)
            
            # 2. One-hot encoding & Padding (maxlen=20)
            vocab_size = 10000
            one_hot_rep = [tf.keras.preprocessing.text.one_hot(cleaned, vocab_size)]
            padded_input = pad_sequences(one_hot_rep, padding='pre', maxlen=20)
            
            # 3. Model Prediction
            preds = model.predict(padded_input)
            pred_class = np.argmax(preds, axis=-1)[0]
            
            # Category Mapping (0: Hate Speech, 1: Offensive, 2: Neither)
            labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
            result = labels.get(pred_class, "Unknown")
            confidence = round(float(np.max(preds)) * 100, 2)

    return render(request, 'index.html', {
        'result': result, 
        'confidence': confidence, 
        'user_input': user_input
    })