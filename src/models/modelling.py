"""
import pandas as pd
import numpy as np 
import pickle 
import yaml

from sklearn.ensemble import RandomForestClassifier

with open("params.yaml","r") as file:
    params = yaml.safe_load(file)

n_estimators = params['modelling']['n_estimators']
max_depth = params['modelling']['max_depth']

train_data = pd.read_csv("data/interim/train_bow.csv")

x_train = train_data.drop(columns=['sentiment']).values
y_train = train_data['sentiment'].values

model = RandomForestClassifier(n_estimators= n_estimators,max_depth = max_depth, random_state=42)
model.fit(x_train, y_train)

pickle.dump(model, open("models/random_forest_model.pkl", "wb"))
"""
import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_params(filepath: str) -> Dict:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
            if not isinstance(params, dict):
                raise ValueError("params.yaml is not formatted correctly.")
            return params
    except Exception as e:
        logging.error(f"Error loading params file: {e}")
        raise

def load_train_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Train data loaded from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading train data: {e}")
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)
        logging.info("Model training completed.")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def save_model(model: Any, filepath: str) -> None:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        n_estimators = params['modelling']['n_estimators']
        max_depth = params['modelling']['max_depth']

        train_data = load_train_data("data/interim/train_bow.csv")
        x_train = train_data.drop(columns=['sentiment']).values
        y_train = train_data['sentiment'].values

        model = train_model(x_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Modelling pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
    