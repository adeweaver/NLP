import pandas as pd
import torch
import nltk
import re
import string

from CustomVocab import CustomVocab
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from collections import Counter
from nltk.tokenize import word_tokenize
from SarcasmModel import SarcasmAnalysisModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


#modify to ensure that only strings are passed to nltk
def tokenize(text):
    if not isinstance(text, str):
        text = str(text)
    return [token.lower() for token in word_tokenize(text)]

class SarcasmDataset(Dataset):
    def __init__(self, data, text_field, label_field):
        self.data = data
        self.text_field = text_field
        self.label_field = label_field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[self.text_field].iloc[index]
        label = self.data[self.label_field].iloc[index]
        return text, label


def load_data(file_path, text_field, label_field, batch_size, max_length, train_frac=0.8):
    #subject to change

    data = pd.read_csv(file_path)
    # data[text_field] = data[text_field].apply(preprocess_text)
    data = data[[text_field, label_field]]

    # Separate positive and negative examples
    data_positive = data[data[label_field] == 1]
    data_negative = data[data[label_field] == 0]

    # Oversample the minority class
    majority_class_size = max(len(data_positive), len(data_negative))
    data_positive_oversampled = data_positive.sample(n=majority_class_size, replace=True, random_state=42)
    data_negative_oversampled = data_negative.sample(n=majority_class_size, replace=True, random_state=42)

    # Combine the oversampled minority class and majority class data
    data_oversampled = pd.concat([data_positive_oversampled, data_negative_oversampled], axis=0)

    # Shuffle the combined data
    data_oversampled = data_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)

    # Replace the original data DataFrame with the data_oversampled DataFrame
    data = data_oversampled

    #-----------------------------------------------------------

    # Print some samples from the raw data
    print("Raw data samples:")
    print(data.head())

    train_data = data.sample(frac=train_frac, random_state=42)
    val_data = data.drop(train_data.index)

    tokens = [token for text in train_data[text_field] for token in tokenize(text)]
    token_counts = Counter(tokens)

    min_freq = 2
    filtered_token_counts = {token: count for token, count in token_counts.items() if count >= min_freq}
    filtered_token_counter = Counter(filtered_token_counts)
    vocab = CustomVocab(filtered_token_counter)

    def collate_fn(batch, vocab, max_length=100):
        texts, labels = zip(*batch)

        encoded_texts = [torch.tensor([vocab.stoi[token] for token in tokenize(text) if token in vocab.stoi]) for text in texts]
        padded_texts = torch.zeros(len(encoded_texts), max_length, dtype=torch.long)

        for i, text in enumerate(encoded_texts):
            text_len = min(len(text), max_length)
            padded_texts[i, :text_len] = text[:text_len]

        labels = torch.tensor(labels, dtype=torch.float)
        return padded_texts, labels

    train_dataset = SarcasmDataset(train_data, text_field, label_field)
    val_dataset = SarcasmDataset(val_data, text_field, label_field)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda b: collate_fn(b, vocab, max_length))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda b: collate_fn(b, vocab, max_length))

    # Print some samples from tokenized, encoded data, and data loaders
    print("\nTokenized data samples:")
    for i, row in train_data.head().iterrows():
        print(tokenize(row[text_field]))

    print("\nEncoded data samples:")
    for i, row in train_data.head().iterrows():
        print([vocab.stoi[token] for token in tokenize(row[text_field]) if token in vocab.stoi])

    print("\nData loader samples:")
    for i, (texts, labels) in enumerate(train_dataloader):
        print("Batch", i+1)
        print("Texts:", texts)
        print("Labels:", labels)
        if i >= 2:
            break

    return train_dataloader, val_dataloader, vocab, data


def preprocess_text(text_field):
    # check if the input is a series object
    if isinstance(text_field, pd.Series):
        # apply the function to each element in the series object
        return text_field.apply(preprocess_text)
    else:
        # check if the input is a string
        if isinstance(text_field, str):
            # convert to lowercase
            text_field = text_field.lower()
            # remove URLs
            text_field = re.sub(r'http\S+', '', text_field)
            # remove non-alphabetic characters
            text_field = re.sub('[^a-z]+', ' ', text_field)
            # remove extra spaces
            text_field = re.sub(' +', ' ', text_field)
            # strip leading and trailing spaces
            text_field = text_field.strip()
        return text_field
    