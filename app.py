from flask import Flask, request, render_template, jsonify
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
import os

application = Flask(__name__)
app = application

@app.route('/')
def index():
    logging.info("Home page accessed")
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect form data
            form_data = {
                'gate_fidelity': float(request.form.get('gate_fidelity')),
                'coherence_time': float(request.form.get('coherence_time')),
                'temperature': float(request.form.get('temperature')),
                'gate_count': int(request.form.get('gate_count')),
                'circuit_depth': int(request.form.get('circuit_depth')),
                'qubit_connectivity': float(request.form.get('qubit_connectivity')),
                'readout_fidelity': float(request.form.get('readout_fidelity')),
                'crosstalk_level': float(request.form.get('crosstalk_level'))
            }
            
            logging.info(f"Received form data: {form_data}")
            
            data = CustomData(**form_data)
            pred_df = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            prediction, probability, impact = predict_pipeline.predict(pred_df)
            
            logging.info(f"Final results - Prediction: {prediction}, Probability: {probability:.4f}")
            
            # Determine risk level based on probability
            if probability > 0.7:
                risk_level = 'High'
            elif probability > 0.4:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Prepare results for template
            result = {
                'prediction': impact['prediction'],
                'probability': impact['probability'],
                'confidence': impact['confidence_level'],
                'net_impact': impact['net_impact'],
                'recommendation': impact['recommendation'],
                'expected_benefit': impact['expected_benefit'],
                'expected_cost': impact['expected_cost'],
                'roi_potential': impact['roi_potential'],
                'risk_level': risk_level
            }
            
            logging.info(f"Business impact calculated successfully")
            
            return render_template('home.html', results=result)
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return render_template('home.html', error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)