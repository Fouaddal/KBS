import numpy as np
import pandas as pd


class StatisticsAnalyzer:

    @staticmethod
    def summary(df: pd.DataFrame):

        return {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "numeric_summary": df.describe().to_dict()
        }

    @staticmethod
    def correlation_matrix(df: pd.DataFrame):

        return df.corr(numeric_only=True)