
"""
import pandas as pd
import numpy as np  
import os
import yaml

from sklearn.feature_extraction.text import CountVectorizer

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

max_features = params['feature_eng']['max_features']    

# Load processed train and test data
train_data = pd.read_csv("data/processed/train.csv").dropna(subset=['content'])
test_data = pd.read_csv("data/processed/test.csv").dropna(subset=['content'])

# Extract features and labels from train and test data
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data and transform it to feature vectors
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer (do not fit again)
X_test_bow = vectorizer.transform(X_test)
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['sentiment'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['sentiment'] = y_test   

# Save the processed feature data to CSV files
os.makedirs("data/interim", exist_ok=True)  # Ensure the directory exists
train_df.to_csv("data/interim/train_bow.csv", index=False)
test_df.to_csv("data/interim/test_bow.csv", index=False)
"""
import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_params(filepath: str) -> dict:
    try:
        with open(filepath, 'r') as file:
            params = yaml.safe_load(file)
            if not isinstance(params, dict):
                raise ValueError("params.yaml is not formatted correctly.")
            return params
    except Exception as e:
        logging.error(f"Error loading params file: {e}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info("Train and test data loaded successfully.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading train/test data: {e}")
        raise

def extract_features(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['sentiment'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['sentiment'] = y_test

        logging.info("Feature extraction completed.")
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error during feature extraction: {e}")
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train_bow.csv")
        test_path = os.path.join(output_dir, "test_bow.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logging.info(f"Processed feature data saved to {output_dir}.")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise

def main() -> None:
    try:
        params = load_params('params.yaml')
        max_features = params['feature_eng']['max_features']
        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        train_df, test_df = extract_features(train_data, test_data, max_features)
        save_data(train_df, test_df, "data/interim")
        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
