from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import os

app = Flask(__name__)

# Load model once at startup
model_path = os.path.join(os.path.dirname(__file__), 'difficulty_ai_weights_with_data_preprocessing2.joblib')
model = load(model_path)

@app.route('/predict', methods=['POST'])
def predict_difficulty():
    data = request.json
    try:
        x_value = pd.DataFrame([{
            'currentDifficulty': data['currentDifficulty'],
            'currentPlayerLives': data['currentPlayerLives'],
            'levelsBeat': data['levelsBeat'],
            'playerLifeTimer': data['playerLifeTimer'],
            'totalEnemiesKilled': data['totalEnemiesKilled'],
            'totalPoints': data['totalPoints']
        }])

        prediction = model.predict(x_value)[0]
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return 'ML API is live'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
