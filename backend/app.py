from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
try:
    model = joblib.load('models/Cardiovascular_disease_model.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    # This serves as the ML Dashboard / Insights page
    return render_template('index.html')

@app.route('/diagnostic')
def diagnostic():
    # This serves as the actual Prediction page
    return render_template('diagnostic.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded.'})
    
    try:
        data = request.form
        age = float(data['age'])
        gender = int(data['gender'])
        height = float(data['height'])
        weight = float(data['weight'])
        ap_hi = float(data['ap_hi'])
        ap_lo = float(data['ap_lo'])
        cholesterol = int(data['cholesterol'])
        gluc = int(data['gluc'])
        smoke = int(data.get('smoke', 0))
        alco = int(data.get('alco', 0))
        active = int(data['active'])

        # Calculate Derived Features required by your trained model
        bmi = weight / ((height/100)**2)
        pp = ap_hi - ap_lo

        # Prepare feature array in the exact order of your training notebook
        features = np.array([[age, gender, height, weight, bmi, ap_hi, ap_lo, 
                              cholesterol, gluc, smoke, alco, active, pp]])

        prob = model.predict_proba(features)[0][1]
        result = 1 if prob > 0.5 else 0

        return jsonify({
            'result': result,
            'prob': round(float(prob) * 100, 2),
            'bmi': round(bmi, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()