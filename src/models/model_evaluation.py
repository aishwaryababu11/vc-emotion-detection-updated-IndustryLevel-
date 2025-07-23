"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import pandas as pd
import pickle   
import json

model  = pickle.load(open("models/random_forest_model.pkl", "rb"))
test_data = pd.read_csv("data/interim/test_bow.csv")

X_test = test_data.drop(columns=['sentiment']).values
y_test = test_data['sentiment'].values  

y_pred = model.predict(X_test)

metrics_dict = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred)
}

with open("reports/metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)
"""

import pandas as pd
import pickle
import json
import logging
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_model(model_path: str) -> Any:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_test_data(test_data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(test_data_path)
        logging.info(f"Test data loaded from {test_data_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise

def evaluate_model(model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
    try:
        X_test = test_data.drop(columns=['sentiment']).values
        y_test = test_data['sentiment'].values
        y_pred = model.predict(X_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info("Model evaluation completed.")
        return metrics_dict
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

def main() -> None:
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_bow.csv")
        metrics = evaluate_model(model, test_data)
        save_metrics(metrics, "reports/metrics.json")
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()