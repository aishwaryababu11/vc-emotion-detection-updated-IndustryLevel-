

"""
import os
import re
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    # Lemmatize each word in the text
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    # Remove stop words from the text
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    # Remove all digits from the text
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    # Convert all words in the text to lowercase
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def removing_punctuations(text):
    # Remove punctuations and extra whitespace from the text
    text = re.sub('[%s]' % re.escape(""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~""), ' ', text) #i have changed 3 " to 2 for commenting the code
    text = text.replace('؛', "", )
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text):
    # Remove URLs from the text
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    # Set text to NaN if sentence has fewer than 3 words
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    # Apply all preprocessing steps to the 'content' column of the DataFrame
    df.content = df.content.apply(lambda content: lower_case(content))
    df.content = df.content.apply(lambda content: remove_stop_words(content))
    df.content = df.content.apply(lambda content: removing_numbers(content))
    df.content = df.content.apply(lambda content: removing_punctuations(content))
    df.content = df.content.apply(lambda content: removing_urls(content))
    df.content = df.content.apply(lambda content: lemmatization(content))
    return df

def normalized_sentence(sentence):
    # Apply all preprocessing steps to a single sentence
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = removing_numbers(sentence)
    sentence = removing_punctuations(sentence)
    sentence = removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence

# Load raw train and test data
train_data = pd.read_csv("data/raw/train.csv")
test_data = pd.read_csv("data/raw/test.csv")

# Normalize train and test data
train_data = normalize_text(train_data)
test_data = normalize_text(test_data)

# Save processed data to CSV files
os.makedirs("data/processed", exist_ok=True)  # Ensure the directory exists
train_data.to_csv("data/processed/train.csv", index=False)
test_data.to_csv("data/processed/test.csv", index=False)

"""
import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any, Callable
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def download_nltk_resources() -> None:
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        logging.info("NLTK resources downloaded successfully.")
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {e}")
        raise

def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in str(text).split() if word not in stop_words])

def removing_numbers(text: str) -> str:
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text: str) -> str:
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    df['content'] = df['content'].apply(lambda x: x if len(str(x).split()) >= 3 else np.nan)
    return df

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        df = remove_small_sentences(df)
        logging.info("Text normalization completed.")
        return df
    except Exception as e:
        logging.error(f"Error during text normalization: {e}")
        raise

def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded data from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logging.info(f"Saved processed data to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")
        raise

def main() -> None:
    try:
        download_nltk_resources()
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")

        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)

        save_data(train_data, "data/processed/train.csv")
        save_data(test_data, "data/processed/test.csv")
        logging.info("Data preprocessing pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()