from sklearn.preprocessing import LabelEncoder
import pandas as pd


class EncodingProcessor:

    @staticmethod
    def label_encoding(series):
        encoder = LabelEncoder()
        return encoder.fit_transform(series)

    @staticmethod
    def one_hot_encoding(df, columns):
        return pd.get_dummies(df, columns=columns)