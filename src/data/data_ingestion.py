"""
import numpy as np
import pandas as pd
import os
import yaml

from sklearn.model_selection import train_test_split

with open("params.yaml","r") as file:
    params = yaml.safe_load(file)

test_size = params['data_ingestion']['test_size'] 

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

df.head()

df.drop(columns=['tweet_id'],inplace=True)

final_df = df[df['sentiment'].isin(['happiness','sadness'])]

final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)

train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

os.makedirs('data/raw', exist_ok=True)

train_data.to_csv('data/raw/train.csv', index=False)
test_data.to_csv('data/raw/test.csv', index=False)
"""
import numpy as np
import pandas as pd
import os
import yaml
import logging
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

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

def load_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=['tweet_id'])
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logging.info("Data preprocessing completed.")
        return final_df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_and_save_data(df: pd.DataFrame, test_size: float, output_dir: str) -> Tuple[str, str]:
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Train and test data saved to {output_dir}.")
        return train_path, test_path
    except Exception as e:
        logging.error(f"Error during train/test split or saving: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        test_size = params['data_ingestion']['test_size']
        df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        split_and_save_data(final_df, test_size, 'data/raw')
        logging.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()