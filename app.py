from flask import Flask, request, jsonify, render_template
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Inisialisasi Flask app
app = Flask(__name__)

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./model')
tokenizer = DistilBertTokenizer.from_pretrained('./tokenizer')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fungsi untuk melakukan prediksi
def predict_tweet(text, model, tokenizer, max_len, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 0:
        return "Not suicidal tweet"
    else:
        return "Potential suicidal tweet"

# Endpoint untuk halaman awal
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint untuk melakukan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['text']
    max_len = 160
    result = predict_tweet(input_text, model, tokenizer, max_len, device)
    return jsonify({'prediction': result})

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
