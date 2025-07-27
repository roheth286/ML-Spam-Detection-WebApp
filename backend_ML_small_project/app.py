from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS  # Import CORS
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests
api = Api(app)

# Set up absolute path for model loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
model_path = os.path.join(BASE_DIR, "model", "spam_classifier1.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "tfidf_vectorizer1.pkl")

# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

class SpamClassifier(Resource):
    def post(self):
        try:
            data = request.get_json()
            if 'text' not in data:
                return jsonify({"error": "Missing 'text' field in request"}), 400

            text = [data['text']]
            text_transformed = vectorizer.transform(text)
            prediction = model.predict(text_transformed)[0]

            return jsonify({"prediction": "spam" if prediction == 1 else "ham"})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Add resource to API
api.add_resource(SpamClassifier, "/predict")

@app.route("/example", methods=["GET"])
def example():
    return jsonify({"message": "Hello, this is the example endpoint!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Allow access from other devices
