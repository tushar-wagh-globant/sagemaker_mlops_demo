import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--n-estimators', type=int, default=10)
    
    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args, _ = parser.parse_known_args()

    print("Reading data")
    # SageMaker uploads the training file to the 'train' directory
    train_file = os.path.join(args.train, "train.csv")
    train_df = pd.read_csv(train_file)

    # Separate features and labels (Assuming last column is label)
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    print("Training model")
    model = RandomForestClassifier(n_estimators=args.n_estimators)
    model.fit(X_train, y_train)

    print("Saving model")
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))