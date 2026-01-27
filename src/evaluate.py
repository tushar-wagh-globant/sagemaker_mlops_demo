import argparse
import os
import json
import tarfile
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

def load_test_data(test_path):
    # Robustly find the CSV file if a directory is passed
    if os.path.isdir(test_path):
        csv_files = glob.glob(os.path.join(test_path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {test_path}")
        test_path = csv_files[0]
        print(f"✅ Found test file: {test_path}")

    print(f"Loading test data from {test_path}...")
    # Using header=None is risky if your data has headers. 
    # Since preprocess.py writes with headers (to_csv(index=False)), we should usually use default (header='infer')
    # Use header=0 to treat first row as header.
    test_df = pd.read_csv(test_path)
    
    # Check if 'quality' column exists (target), otherwise assume last column
    if 'quality' in test_df.columns:
        X_test = test_df.drop('quality', axis=1).values
        y_test = test_df['quality'].values
    else:
        # Fallback to index-based splitting
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
    
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
    }
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return metrics, cm, report, y_pred

def main():
    parser = argparse.ArgumentParser()
    # Default to directories where SageMaker mounts inputs
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--test-dir', type=str, default='/opt/ml/processing/test')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/evaluation')
    
    args = parser.parse_args()
    
    # --- 1. Extract and Load Model ---
    model_dir = args.model_dir
    tar_path = os.path.join(model_dir, "model.tar.gz")
    
    # Extract tarball if it exists (Standard SageMaker behavior)
    if os.path.exists(tar_path):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=model_dir)
            
    # Look for the model file (joblib preferred, fallback to pkl)
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model.pkl")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Could not find model.joblib or model.pkl in {model_dir}")

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # --- 2. Load Data ---
    X_test, y_test = load_test_data(args.test_dir)
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # --- 3. Evaluate ---
    print("Evaluating model...")
    metrics, cm, report, y_pred = evaluate_model(model, X_test, y_test)
    
    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    
    # --- 4. Save Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Structure specifically for Pipeline JsonGet
    evaluation_output = {
        'metrics': {
            'accuracy': {'value': metrics['accuracy']},
            'precision': {'value': metrics['precision']},
            'recall': {'value': metrics['recall']},
            'f1': {'value': metrics['f1_score']}
        }
    }
    
    evaluation_path = os.path.join(args.output_dir, 'evaluation.json')
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_output, f, indent=2)
    
    print(f"\nEvaluation results saved to {evaluation_path}")

if __name__ == "__main__":
    main()