from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Optional form-based UI

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form or JSON
        if request.is_json:
            data = request.get_json()
            features = np.array([list(data.values())])
        else:
            features = [float(x) for x in request.form.values()]
            features = np.array([features])

        prediction = model.predict(features)
        return jsonify({'prediction': prediction[0][0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
