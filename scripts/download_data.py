import pandas as pd
import os


def download_wine_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    print(f"Downloading wine quality dataset from {url}...")
    df = pd.read_csv(url, sep=';')
    
    os.makedirs('data', exist_ok=True)
    output_path = 'data/wine-quality.csv'
    
    df.to_csv(output_path, sep=';', index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nColumn info:")
    print(df.info())
    
    return df


if __name__ == "__main__":
    download_wine_data()
