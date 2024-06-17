import os
import pickle
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import logging
from google.cloud import storage
from flask import Flask, request, jsonify

app = Flask(__name__)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress Hugging Face Transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)

model_local_path = './tmp/saved_model'
tokenizer_local_path = './tmp/saved_model'

# Load model
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_local_path)
model = TFRobertaForSequenceClassification.from_pretrained(model_local_path)

category_to_id_path = './tmp/category_to_id.pkl'
id_to_category_path = './tmp/id_to_category.pkl'

# Load category mappings
with open(category_to_id_path, 'rb') as f:
    category_to_id = pickle.load(f)

with open(id_to_category_path, 'rb') as f:
    id_to_category = pickle.load(f)

def predict_category(note):
    inputs = tokenizer(note, return_tensors='tf', truncation=True, padding=True)
    logits = model(inputs).logits
    predicted_class_id = tf.argmax(logits, axis=1).numpy()[0]
    return id_to_category[predicted_class_id]

@app.route('/predict', methods=['GET'])
def success():
    return jsonify({'message': 'Berhasil!'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'describe' not in data:
            return jsonify({'error': 'Invalid input data: Missing "describe"'}), 400
        
        describe = data['describe']

        prediction = predict_category(str(describe))
        return jsonify({'category': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)