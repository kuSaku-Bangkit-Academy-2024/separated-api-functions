from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import logging
from google.cloud import storage
from flask import Flask, request, jsonify

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_path = './tmp/model.h5'
scaler_path = './tmp/scaler.pkl'

model = load_model(model_path)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

def predict_expenses(input_data):
    """
    Predicts expenses based on the input income using a pre-trained model.

    Parameters:
    input_data (pd.DataFrame): DataFrame containing the input data with a column named 'Income'.
    model_path (str): Path to the saved model.
    scaler_path (str): Path to the saved scaler (optional).

    Returns:
    pd.DataFrame: DataFrame containing the predicted expenses.
    """
    print(input_data)
    normalized_income = scaler.transform(input_data[['Income']])
    
    predictions = model.predict(normalized_income)
    
    features = ['Food', 'Household', 'Education', 'Health', 'Transportation', 'Apparel', 'Social Life', 'Entertainment', 'Other']
    
    predictions_df = pd.DataFrame(predictions, columns=features)
    
    result_df = input_data.copy()
    result_df[features] = predictions_df
    
    return result_df
@app.route('/', methods=['GET'])
def success():
    return jsonify({'message': 'Berhasil!'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'Income' not in data:
            return jsonify({'error': 'Invalid input data: Missing "Income"'}), 400
        
        Income =  pd.DataFrame(data)

        prediction = predict_expenses(Income)
        print(prediction.to_dict())
        return jsonify(prediction.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)