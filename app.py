from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# 🌍 Global model/tokenizer yüklenir
try:
    print("🚀 Model yükleniyor...")
    model = load_model("model.h5")
    print("✅ model.h5 yüklendi.")
except Exception as e:
    print("❌ model yükleme hatası:", e)
    model = None

try:
    print("🚀 Tokenizer yükleniyor...")
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    print("✅ tokenizer.json yüklendi.")
except Exception as e:
    print("❌ tokenizer yükleme hatası:", e)
    tokenizer = None

maxlen = 100
explainer = LimeTextExplainer(class_names=["negatif", "pozitif"])

def predict_texts(texts):
    if model is None or tokenizer is None:
        raise Exception("Model veya tokenizer yüklenemedi.")
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen)
    preds = model.predict(padded)
    return np.hstack([1 - preds, preds])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        print("📝 Prediction isteği:", text)
        output = predict_texts([text])
        score = float(output[0][1])
        return jsonify({'prediction': score})
    except Exception as e:
        print("❌ predict hatası:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/lime', methods=['POST'])
def lime():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        if len(text.strip().split()) < 3:
            return jsonify({'error': 'Yorum çok kısa, en az 3 kelime girin.'}), 400

        print("🧠 LIME başlatıldı. Yorum:", text)

        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_texts,
            labels=[1],
            num_features=10,
            num_samples=250
        )

        explanation = dict(exp.as_list(label=1))
        print("✅ Açıklama üretildi:", explanation)
        return jsonify({'explanation': explanation})
    except Exception as e:
        print("❌ LIME hatası:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
