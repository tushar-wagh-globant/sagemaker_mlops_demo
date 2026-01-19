import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import train_model
from sklearn.datasets import make_classification


def test_train_model():
    X_train, y_train = make_classification(
        n_samples=100,
        n_features=11,
        n_classes=2,
        random_state=42
    )
    
    hyperparameters = {
        'n_estimators': 10,
        'max_depth': 5,
        'random_state': 42
    }
    
    model = train_model(X_train, y_train, hyperparameters)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
    predictions = model.predict(X_train[:5])
    assert len(predictions) == 5
    assert all(p in [0, 1] for p in predictions)


def test_model_accuracy():
    X_train, y_train = make_classification(
        n_samples=200,
        n_features=11,
        n_classes=2,
        random_state=42
    )
    
    hyperparameters = {
        'n_estimators': 50,
        'max_depth': 10,
        'random_state': 42
    }
    
    model = train_model(X_train, y_train, hyperparameters)
    
    accuracy = model.score(X_train, y_train)
    
    assert accuracy > 0.5, "Model should perform better than random guessing"
    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
