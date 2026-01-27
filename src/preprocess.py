import argparse
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(input_path):
    print(f"Searching for data at: {input_path}")
    
    # If path is a folder, find the CSV inside
    if os.path.isdir(input_path):
        csv_files = glob.glob(os.path.join(input_path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {input_path}")
        input_path = csv_files[0]
        
    print(f"Loading file: {input_path}")
    return pd.read_csv(input_path, sep=';')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--train-output', type=str, default='/opt/ml/processing/output/train')
    parser.add_argument('--test-output', type=str, default='/opt/ml/processing/output/test')
    
    args = parser.parse_args()
    
    df = load_data(args.input_data)
    
    # Basic cleanup
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    print("Splitting data...")
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)
    
    train.to_csv(os.path.join(args.train_output, 'train.csv'), index=False)
    test.to_csv(os.path.join(args.test_output, 'test.csv'), index=False)
    print("Done.")

if __name__ == '__main__':
    main()