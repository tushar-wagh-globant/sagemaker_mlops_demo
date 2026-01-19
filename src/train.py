import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


def load_training_data(train_path):
    train_df = pd.read_csv(train_path, header=None)
    
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    
    return X_train, y_train


def train_model(X_train, y_train, hyperparameters):
    print("Training Random Forest Classifier...")
    print(f"Hyperparameters: {hyperparameters}")
    
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    
    return model


def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=2)
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    parser.add_argument('--random-state', type=int, default=42)
    
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--local', action='store_true', help='Run in local mode for testing')
    
    args = parser.parse_args()
    
    hyperparameters = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'random_state': args.random_state,
    }
    
    train_file = os.path.join(args.train, 'train.csv')
    
    if not os.path.exists(train_file):
        raise ValueError(f"Training file not found: {train_file}")
    
    print(f"Loading training data from {train_file}...")
    X_train, y_train = load_training_data(train_file)
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    model = train_model(X_train, y_train, hyperparameters)
    
    train_accuracy = model.score(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    save_model(model, args.model_dir)
    
    metrics = {
        'train_accuracy': train_accuracy,
        'hyperparameters': hyperparameters
    }
    
    metrics_path = os.path.join(args.model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
