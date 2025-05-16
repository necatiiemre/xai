from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

model = load_model("model.h5")
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())

maxlen = 100
explainer = LimeTextExplainer(class_names=["negatif", "pozitif"])

def predict_texts(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen)
    preds = model.predict(padded)
    return np.hstack([1 - preds, preds])

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        print("📝 Gelen yorum:", text)
        prob = float(predict_texts([text])[0][1])
        return jsonify({'prediction': prob})
    
    except Exception as e:
        print("❌ Hata:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/lime', methods=['POST'])
def lime():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')

        print("🧠 LIME başlatıldı. Yorum:", text)

        result = predict_texts([text])
        print("🔍 predict_texts çıktısı:", result, "şekil:", result.shape)

        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_texts,
            labels=[1],
            num_features=10
        )

        explanation = dict(exp.as_list(label=1))
        print("✅ Açıklama üretildi.")
        return jsonify({'explanation': explanation})

    except Exception as e:
        print("❌ LIME hatası:", e)
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
