from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [[data['square_footage'], data['bedrooms'], data['bathrooms']]]
    prediction = model.predict(features)
    return jsonify({'predicted_price': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
