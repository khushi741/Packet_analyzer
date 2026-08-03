import sys
import json
import joblib
import pandas as pd

def load_model_and_predict(json_path_or_str: str):
    try:
        if json_path_or_str.endswith('.json'):
            with open(json_path_or_str, 'r') as f:
                data = json.load(f)
        else:
            data = json.loads(json_path_or_str)
    except Exception as e:
        print(f"Error loading JSON input: {e}")
        sys.exit(1)
        
    model_path = "ml/models/traffic_classifier.joblib"
    cols_path = "ml/models/feature_columns.joblib"
    
    try:
        model = joblib.load(model_path)
        feature_cols = joblib.load(cols_path)
    except Exception as e:
        print(f"Error loading model files: {e}. Make sure to train the model first.")
        sys.exit(1)
        
    # Standardize input keys to list of features
    input_data = {}
    for col in feature_cols:
        input_data[col] = [data.get(col, 0.0)]
        
    df_input = pd.DataFrame(input_data)
    
    prediction = model.predict(df_input)[0]
    probs = model.predict_proba(df_input)[0]
    confidence = max(probs)
    
    result = {
        "prediction": prediction,
        "confidence": float(confidence)
    }
    
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <json_file_path_or_string>")
        sys.exit(1)
    load_model_and_predict(sys.argv[1])
