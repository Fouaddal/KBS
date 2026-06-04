import pandas as pd


class Helpers:

    @staticmethod
    def check_missing(df: pd.DataFrame):

        return df.isnull().sum().sum() > 0

    @staticmethod
    def split_features_target(df, target_column):

        X = df.drop(columns=[target_column])
        y = df[target_column]

        return X, y