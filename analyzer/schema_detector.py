import pandas as pd


class SchemaDetector:

    @staticmethod
    def detect_types(df: pd.DataFrame):

        return {
            "numerical": list(df.select_dtypes(include=["int64", "float64"]).columns),
            "categorical": list(df.select_dtypes(include=["object"]).columns),
            "datetime": list(df.select_dtypes(include=["datetime64"]).columns)
        }

    @staticmethod
    def has_text_features(df: pd.DataFrame):

        return any(df[col].astype(str).str.len().mean() > 20
                   for col in df.select_dtypes(include=["object"]).columns)