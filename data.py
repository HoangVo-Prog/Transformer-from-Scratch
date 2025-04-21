import pandas as pd
import pickle
from datasets import load_dataset  


# Load the dataset from Hugging Face
def save_data(train_data, valid_data, test_data, filename='dataset.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((train_data, valid_data, test_data), f)

# Load the data
def load_data(filename='dataset.pkl'):
    with open(filename, 'rb') as f:
        train_data, valid_data, test_data = pickle.load(f)
    return train_data, valid_data, test_data


if __name__ == "__main__":
    # Load the dataset from Hugging Face
    ds = load_dataset("thainq107/iwslt2015-en-vi")
    train_data, valid_data, test_data = ds["train"], ds["validation"], ds["test"]
    save_data(train_data, valid_data, test_data)
