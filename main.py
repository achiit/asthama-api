from flask import Flask, request, jsonify
import joblib
import traceback
import numpy as np
from flask.json import JSONEncoder

app = Flask(__name__)

def predicter(g, p, q, r, s, actual_pefr):
    try:
        model = joblib.load('decision_tree_model.joblib')
        prediction = model.predict([[g, p, q, r, s]])
        predicted_pefr = prediction[0]
        perpefr = (actual_pefr / predicted_pefr) * 100
        
        if perpefr >= 80:
            re = 'SAFE'
        elif perpefr >= 50:
            re = 'MODERATE'
        else:
            re = 'RISK'
        
        return {
            'status': re,
            'predicted_pefr': float(predicted_pefr),
            'actual_pefr': float(actual_pefr),
            'pefr_percentage': float((perpefr // 100) * 10)
        }
    except Exception as e:
        print(f"An error occurred in predicter: {str(e)}")
        print(traceback.format_exc())
        return {'error': str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        g = float(data['gender'])
        p = float(data['temperature'])
        q = float(data['humidity'])
        r = float(data['pm25'])
        s = float(data['pm10'])
        actual_pefr = float(data['actual_pefr'])

        result = predicter(g, p, q, r, s, actual_pefr)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)