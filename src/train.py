import argparse
import os
import joblib
import pandas as pd
import shutil  # ✅ Added for copying files
from sklearn.ensemble import RandomForestClassifier

def model_fn(model_dir):
    """Load model from the model_dir."""
    print("Loading model.")
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=10)
    
    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args, _ = parser.parse_known_args()

    print("Reading data")
    # SageMaker uploads the training file to the 'train' directory
    train_file = os.path.join(args.train, "train.csv")
    train_df = pd.read_csv(train_file)

    # Separate features and labels
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    print("Training model")
    model = RandomForestClassifier(n_estimators=args.n_estimators)
    model.fit(X_train, y_train)

    print("Saving model")
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    # ------------------------------------------------------------------
    # ✅ CRITICAL FIX: Repack inference.py into the model artifact
    # ------------------------------------------------------------------
    # SageMaker expects custom entry points to be in a 'code' folder 
    # inside model.tar.gz for automatic detection.
    
    inference_code_dir = os.path.join(args.model_dir, "code")
    os.makedirs(inference_code_dir, exist_ok=True)
    
    # Since we set source_dir="src" in the estimator, inference.py is available
    # in the current working directory during training.
    if os.path.exists("inference.py"):
        print("📦 Packing inference.py into model artifact...")
        shutil.copy("inference.py", os.path.join(inference_code_dir, "inference.py"))
    else:
        print("⚠️ Warning: inference.py not found in training environment!")
        # Attempt to list files to debug if it's missing
        print(f"Current directory contents: {os.listdir('.')}")