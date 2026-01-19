import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib


def load_test_data(test_path):
    test_df = pd.read_csv(test_path, header=None)
    
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
    
    parser.add_argument('--model-path', type=str, default='/opt/ml/processing/model/model.pkl')
    parser.add_argument('--test-path', type=str, default='/opt/ml/processing/test/test.csv')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/evaluation')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model = joblib.load(args.model_path)
    
    print(f"Loading test data from {args.test_path}...")
    X_test, y_test = load_test_data(args.test_path)
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    print("Evaluating model...")
    metrics, cm, report, y_pred = evaluate_model(model, X_test, y_test)
    
    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    
    print("\nClassification Report:")
    print(report)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    os.makedirs(args.output_path, exist_ok=True)
    
    evaluation_output = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    evaluation_path = os.path.join(args.output_path, 'evaluation.json')
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_output, f, indent=2)
    
    print(f"\nEvaluation results saved to {evaluation_path}")


if __name__ == "__main__":
    main()
