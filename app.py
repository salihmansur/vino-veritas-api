from flask import Flask, request, jsonify
import joblib
import numpy as np

# 1. Set up Flask: Create a basic Flask application.
app = Flask(__name__)

# 2. Load the Model (and scaler) saved in Part 1.
model = joblib.load("vino_veritas_model.pkl")
scaler = joblib.load("scaler.pkl")

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
