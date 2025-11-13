import os
import sys
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.random import random_circuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logging.warning("Qiskit not available. Will use fallback data generation.")

import random

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def generate_quantum_data(self, n_samples=15000):
        """
        Generate synthetic quantum computing data using Qiskit
        """
        try:
            logging.info("Generating quantum computing dataset")
            
            data = []
            
            for i in range(n_samples):
                try:
                    if QISKIT_AVAILABLE:
                        # Generate random Qiskit circuit
                        n_qubits = random.randint(2, 8)
                        depth = random.randint(3, 15)
                        circuit = random_circuit(n_qubits, depth, measure=True, seed=i)
                        
                        # Extract circuit features
                        features = self.extract_circuit_features(circuit)
                        
                        circuit_data = {
                            'gate_count': features['gate_count'],
                            'circuit_depth': features['circuit_depth'],
                        }
                    else:
                        # Fallback without Qiskit
                        circuit_data = {
                            'gate_count': np.random.poisson(50) + 10,
                            'circuit_depth': np.random.poisson(20) + 5,
                        }
                    
                    # Generate hardware parameters
                    circuit_data['gate_fidelity'] = np.random.beta(20, 2) * 0.099 + 0.9
                    circuit_data['coherence_time'] = np.clip(np.random.exponential(30) + 10, 10, 100)
                    circuit_data['temperature'] = np.clip(np.random.gamma(2, 10) + 10, 10, 50)
                    circuit_data['qubit_connectivity'] = np.random.beta(3, 2) * 0.5 + 0.5
                    circuit_data['readout_fidelity'] = np.random.beta(5, 2) * 0.18 + 0.8
                    circuit_data['crosstalk_level'] = np.clip(np.random.exponential(0.03) + 0.01, 0.01, 0.1)
                    
                    data.append(circuit_data)
                
                except Exception as e:
                    # Fallback to random generation
                    fallback = {}
                    fallback['gate_fidelity'] = np.random.beta(20, 2) * 0.099 + 0.9
                    fallback['coherence_time'] = np.clip(np.random.exponential(30) + 10, 10, 100)
                    fallback['temperature'] = np.clip(np.random.gamma(2, 10) + 10, 10, 50)
                    fallback['gate_count'] = np.random.poisson(50) + 10
                    fallback['circuit_depth'] = np.random.poisson(20) + 5
                    fallback['qubit_connectivity'] = np.random.beta(3, 2) * 0.5 + 0.5
                    fallback['readout_fidelity'] = np.random.beta(5, 2) * 0.18 + 0.8
                    fallback['crosstalk_level'] = np.clip(np.random.exponential(0.03) + 0.01, 0.01, 0.1)
                    data.append(fallback)
                
                # Progress indicator
                if (i + 1) % 1000 == 0:
                    logging.info(f"Generated {i + 1}/{n_samples} samples...")
            
            df = pd.DataFrame(data)
            
            # Generate error probability
            error_prob = (
                (1 - df['gate_fidelity']) * 2 +
                (1/df['coherence_time']) * 0.5 +
                (df['temperature'] / 50) * 0.3 +
                (df['gate_count'] / 100) * 0.4 +
                (df['circuit_depth'] / 50) * 0.3 +
                (1 - df['qubit_connectivity']) * 0.2 +
                (1 - df['readout_fidelity']) * 0.5 +
                df['crosstalk_level'] * 2
            )
            
            error_prob += np.random.normal(0, 0.1, len(df))
            error_prob = np.clip(error_prob, 0, 1)
            
            threshold = np.percentile(error_prob, 70)
            df['error_occurred'] = (error_prob > threshold).astype(int)
            
            logging.info(f"Generated {len(df)} samples with {df['error_occurred'].mean():.2%} error rate")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def extract_circuit_features(self, circuit):
        """Extract features from quantum circuit"""
        features = {}
        features['gate_count'] = len(circuit.data)
        features['circuit_depth'] = circuit.depth()
        return features

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # Generate or read data
            df = self.generate_quantum_data(n_samples=15000)
            logging.info('Dataset generated successfully')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)