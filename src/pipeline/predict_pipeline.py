import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, calculate_business_impact
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            logging.info("Loading model and preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Scaling input features")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making prediction")
            prediction = model.predict(data_scaled)
            probability = model.predict_proba(data_scaled)[0][1]
            
            # Ensure proper types
            prediction_val = int(prediction[0])
            probability_val = float(probability)
            
            logging.info(f"Prediction: {prediction_val}, Probability: {probability_val:.4f}")
            
            # Calculate business impact
            impact = calculate_business_impact(prediction_val, probability_val)
            
            return prediction_val, probability_val, impact
            
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {e}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gate_fidelity: float,
                 coherence_time: float,
                 temperature: float,
                 gate_count: int,
                 circuit_depth: int,
                 qubit_connectivity: float,
                 readout_fidelity: float,
                 crosstalk_level: float):
        
        self.gate_fidelity = gate_fidelity
        self.coherence_time = coherence_time
        self.temperature = temperature
        self.gate_count = gate_count
        self.circuit_depth = circuit_depth
        self.qubit_connectivity = qubit_connectivity
        self.readout_fidelity = readout_fidelity
        self.crosstalk_level = crosstalk_level

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gate_fidelity": [self.gate_fidelity],
                "coherence_time": [self.coherence_time],
                "temperature": [self.temperature],
                "gate_count": [self.gate_count],
                "circuit_depth": [self.circuit_depth],
                "qubit_connectivity": [self.qubit_connectivity],
                "readout_fidelity": [self.readout_fidelity],
                "crosstalk_level": [self.crosstalk_level],
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            
            # Add engineered features (same as training)
            df['error_susceptibility'] = (1 - df['gate_fidelity']) * df['gate_count'] / df['coherence_time']
            df['thermal_factor'] = df['temperature'] * df['crosstalk_level']
            df['circuit_complexity'] = df['gate_count'] * df['circuit_depth']
            df['system_quality'] = df['gate_fidelity'] * df['readout_fidelity'] * df['qubit_connectivity']
            
            return df

        except Exception as e:
            raise CustomException(e, sys)