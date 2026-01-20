import sys
import os
from flask import Flask, request, jsonify, render_template

# Add parent directory to path to resolve src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict_email

# Create Flask app with custom template and static folders
app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/details')
def details_page():
    return render_template('details.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.json.get('email')
    if not email:
        return jsonify({"error": "Email text is required"}), 400

    result = predict_email("models/phishing_detector.pkl", "data/preprocessed_data.pkl", email)
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
