# %%
from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load models and preprocessors
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

with open('LogisticRegresion.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Home Page - Choose Prediction Type
@app.route('/')
def index():
    return render_template('home.html')

# Route for Predicting Crop Yield
@app.route('/predict_yield', methods=['GET', 'POST'])
def predict_yield():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        # Create the feature array
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features)
        formatted_prediction = f"{prediction.item():.2f} Hectograms per Hectare"

        return render_template('yield_form.html', prediction=formatted_prediction)
    
    return render_template('yield_form.html')

# Route for Predicting Crop Recommendation
@app.route('/predict_crop', methods=['GET', 'POST'])
def predict_crop():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        crop_label = prediction[0]

        return render_template('crop_form.html', prediction=crop_label)
    
    return render_template('crop_form.html')

if __name__ == '__main__':
    import os
    if 'ipykernel' in os.environ.get('JPY_PARENT_PID', ''):
        app.run(debug=True)
    else:
        app.run(debug=True, use_reloader=False)



