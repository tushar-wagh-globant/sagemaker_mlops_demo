import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


import os
import glob
import pandas as pd
import sys


def load_data(input_path):
    print(f"🕵️‍♀️ Searching for data at: {input_path}")
    
    # 1. If the path explicitly exists (file or folder), use it
    if os.path.exists(input_path):
        if os.path.isfile(input_path):
            print(f"✅ Found exact file: {input_path}")
            return pd.read_csv(input_path, sep=';')
        elif os.path.isdir(input_path):
            # It's a folder, search inside
            search_path = input_path
    else:
        # 2. Path doesn't exist (e.g., .../input/data.csv). 
        # Assume the user meant the PARENT directory.
        print(f"⚠️ Path not found. Checking parent directory...")
        search_path = os.path.dirname(input_path)

    # 3. Find any CSV in the search directory
    print(f"📂 Scanning directory: {search_path}")
    csv_files = glob.glob(os.path.join(search_path, "*.csv"))
    
    if not csv_files:
        # Critical failure: List everything so we can debug
        print(f"❌ No CSV files found in {search_path}. Directory contents:")
        print(os.listdir(search_path))
        sys.exit(1)
        
    # 4. Use the first CSV found
    target_file = csv_files[0]
    print(f"🎉 Auto-detected file: {target_file}")
    
    return pd.read_csv(target_file, sep=';')



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
