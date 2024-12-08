# predict.py
import pickle
from flask import Flask, request, jsonify

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    X = [[data["square_footage"], data["bedrooms"], data["bathrooms"]]]
    prediction = model.predict(X)
    return jsonify({"predicted_price": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

