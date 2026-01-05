from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# 1. Set up Flask: Create a basic Flask application.
app = Flask(__name__)
@app.route("/")
def home():
    return jsonify({"status": "vino-veritas-api running"})
CORS(app)
@app.route("/", methods=["GET"])
def home():
    return "Vino Veritas API is running. Use POST /predict."
# 2. Load the Model (and scaler) saved in Part 1.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "vino_veritas_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# The 11 features in the same order as training
FEATURE_ORDER = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# 3. Create the /predict endpoint that accepts POST with JSON.
@app.route('/predict', methods=['POST'])
def predict():
    # Extract JSON data from request body
    data = request.get_json()

    # Convert JSON to a NumPy array in the same format as training
    features = [data[name] for name in FEATURE_ORDER]
    X = np.array([features])

    # Scale features using the saved scaler
    X_scaled = scaler.transform(X)

    # Use the loaded model to make a prediction
    y_pred = model.predict(X_scaled)[0]
    label = "Good" if y_pred == 1 else "Bad"

    # Return the prediction as JSON
    return jsonify({"prediction": label, "raw_label": int(y_pred)})

# Run the app if this file is executed directly
if __name__ == "__main__":
    app.run(debug=True, port=5001)

