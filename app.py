from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

# 🌍 Global tanımlar
model = None
tokenizer = None
maxlen = 100
explainer = LimeTextExplainer(class_names=["negatif", "pozitif"])

# 🔥 Flask uygulaması
app = Flask(__name__)

# 📦 Model ve tokenizer yalnızca bir kez yüklenir
@app.before_first_request
def load_assets():
    global model, tokenizer
    print("🚀 İlk yükleme başlıyor...")
    model = load_model("model.h5")
    print("✅ model.h5 yüklendi.")
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    print("✅ tokenizer.json yüklendi.")

# ✅ Prediction fonksiyonu (LIME uyumlu)
def predict_texts(texts):
    try:
        print("🧪 predict_texts() çağrıldı. input len:", len(texts))
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=maxlen)
        preds = model.predict(padded)
        output = np.hstack([1 - preds, preds])
        print("📊 predict_texts output shape:", output.shape)
        return output
    except Exception as e:
        print("❌ predict_texts hatası:", e)
        raise

# 🔹 /predict → yalnızca pozitif skor döner
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        print("📝 Prediction isteği. Yorum:", text)
        output = predict_texts([text])
        score = float(output[0][1])
        return jsonify({'prediction': score})
    except Exception as e:
        print("❌ Predict endpoint hatası:", e)
        return jsonify({'error': str(e)}), 500

# 🔹 /lime → yalnızca açıklama döner
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
            num_samples=250  # 🔧 optimize edildi
        )

        explanation = dict(exp.as_list(label=1))
        print("✅ Açıklama üretildi:", explanation)
        return jsonify({'explanation': explanation})
    except Exception as e:
        print("❌ LIME hatası:", e)
        return jsonify({'error': str(e)}), 500

# 🔃 Railway port ayarı
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
