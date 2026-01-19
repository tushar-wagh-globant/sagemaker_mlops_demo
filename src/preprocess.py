import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(input_path):
    df = pd.read_csv(input_path, sep=';')
    return df


def preprocess_data(df, target_column='quality'):
    df[target_column] = df[target_column].apply(lambda x: 1 if x >= 6 else 0)
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return X, y


def feature_engineering(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input/data.csv')
    parser.add_argument('--train-output', type=str, default='/opt/ml/processing/output/train')
    parser.add_argument('--test-output', type=str, default='/opt/ml/processing/output/test')
    parser.add_argument('--model-output', type=str, default='/opt/ml/processing/output/model')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_data(args.input_data)
    print(f"Data shape: {df.shape}")
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    print("Feature engineering...")
    X_scaled, scaler = feature_engineering(X)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)
    os.makedirs(args.model_output, exist_ok=True)
    
    print("Saving processed data...")
    train_df = pd.DataFrame(X_train)
    train_df['target'] = y_train.values
    train_df.to_csv(f"{args.train_output}/train.csv", index=False, header=False)
    
    test_df = pd.DataFrame(X_test)
    test_df['target'] = y_test.values
    test_df.to_csv(f"{args.test_output}/test.csv", index=False, header=False)
    
    joblib.dump(scaler, f"{args.model_output}/scaler.pkl")
    
    print("Preprocessing complete!")
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")


if __name__ == "__main__":
    main()
