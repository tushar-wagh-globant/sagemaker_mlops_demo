import os
import json
import joblib
import numpy as np


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.pkl'))
    return model


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
