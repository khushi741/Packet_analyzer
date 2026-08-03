import os
import unittest
import json
import pandas as pd
import joblib
from ml.generate_synthetic_flows import generate_flows
from ml.build_dataset import build_dataset
from ml.train_model import train_model
from fastapi.testclient import TestClient
from ml.realtime_inference import app

class TestMLPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # 1. Generate test dataset
        generate_flows(num_flows=50)
        build_dataset()
        train_model()
        cls.client = TestClient(app)
        
    def test_dataset_generation(self):
        self.assertTrue(os.path.exists("ml/dataset/synthetic_flows.csv"))
        self.assertTrue(os.path.exists("ml/dataset/traffic_dataset.csv"))
        df = pd.read_csv("ml/dataset/traffic_dataset.csv")
        self.assertGreaterEqual(len(df), 50)
        self.assertIn("mean_packet_size", df.columns)
        
    def test_model_training(self):
        self.assertTrue(os.path.exists("ml/models/traffic_classifier.joblib"))
        self.assertTrue(os.path.exists("ml/models/feature_columns.joblib"))
        
    def test_api_inference_endpoint(self):
        payload = {
            "flow_duration_ms": 1200.0,
            "total_packets": 42,
            "total_bytes": 87000,
            "mean_packet_size": 1280.0,
            "mean_inter_arrival_time_ms": 15.0
        }
        response = self.client.post("/predict-flow", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("is_anomaly", data)
        
if __name__ == "__main__":
    unittest.main()
