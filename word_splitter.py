from string import punctuation
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder

def preprocess(text):
    text = text.lower()
    text = "".join([char for char in text if char not in punctuation])
    text = text.split()
    text = DataFrame({"token": text})
    return text

def shift(data):
    return data[:-1], data[1:]

class Preprocessor:

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)
        self.vocabulary_size = 0

    def fit(self, text):
        text = preprocess(text)
        self.encoder.fit(text)
        return self

    def transform(self, text):
        text = preprocess(text)
        text = self.encoder.transform(text)
        return text

    def make_data(self, text):
        text = self.transform(text)
        X, y = shift(text)
        self.vocabulary_size = X.shape[1]
        return X, y