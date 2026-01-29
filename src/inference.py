import os
import json
import joblib
import numpy as np


def model_fn(model_dir):
    import os
    # Try different model file names
    for filename in ['model.joblib', 'model.pkl', 'model']:
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            return joblib.load(model_path)
    
    # If nothing found, show what's actually there
    files = os.listdir(model_dir)
    raise FileNotFoundError(f"No model file found in {model_dir}. Found: {files}")


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        if 'instances' in input_data:
            return np.array(input_data['instances'])
        else:
            return np.array(input_data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    return {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist()
    }


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
