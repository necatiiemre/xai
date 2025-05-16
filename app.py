from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

# Load model and tokenizer
model = load_model("model.h5")
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())  

maxlen = 100  # or whatever you used

explainer = LimeTextExplainer(class_names=["negatif", "pozitif"])

def predict_texts(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen)
    preds = model.predict(padded)
    return np.hstack([1 - preds, preds])

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    prob = float(predict_texts([text])[0][1])
    return jsonify({'prediction': prob})

@app.route('/lime', methods=['POST'])
def lime():
    data = request.get_json()
    text = data.get('text', '')
    exp = explainer.explain_instance(text, predict_texts, labels=[1], num_features=10)
    explanation = dict(exp.as_list(label=1))
    return jsonify({'explanation': explanation})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
