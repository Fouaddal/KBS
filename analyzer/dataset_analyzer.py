# =====================================================
# analyzer/dataset_analyzer.py
# =====================================================

import pandas as pd

from scipy.stats import skew

from engine.facts import DatasetFact

from analyzer.feature_analyzer import (
    FeatureAnalyzer
)


class DatasetAnalyzer:

    def __init__(
        self,
        dataframe,
        target_column=None
    ):

        self.df = dataframe.copy()

        self.target_column = target_column

    # =====================================================
    # OUTLIER DETECTION
    # =====================================================

    def detect_outliers(self, column):

        q1 = self.df[column].quantile(0.25)

        q3 = self.df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)

        upper = q3 + (1.5 * iqr)

        outliers = self.df[

            (self.df[column] < lower) |

            (self.df[column] > upper)
        ]

        return len(outliers) > 0

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyze(self):

        # =================================================
        # NUMERICAL COLUMNS
        # =================================================

        numerical_columns = list(

            self.df.select_dtypes(

                include=['int64', 'float64']

            ).columns
        )

        # =================================================
        # CATEGORICAL COLUMNS
        # =================================================

        categorical_columns = list(

            self.df.select_dtypes(

                include=['object']

            ).columns
        )

        # =================================================
        # MISSING VALUES
        # =================================================

        missing_columns = list(

            self.df.columns[
                self.df.isnull().any()
            ]
        )

        has_missing_values = (
            len(missing_columns) > 0
        )

        # =================================================
        # HIGH DIMENSIONALITY
        # =================================================

        high_dimensionality = (
            self.df.shape[1] > 20
        )

        # =================================================
        # SKEWED COLUMNS
        # =================================================

        skewed_columns = []

        for col in numerical_columns:

            try:

                skew_value = skew(
                    self.df[col].dropna()
                )

                if abs(skew_value) > 1:

                    skewed_columns.append(col)

            except Exception:
                pass

        # =================================================
        # OUTLIER COLUMNS
        # =================================================

        has_outliers = False

        outlier_columns = []

        for col in numerical_columns:

            try:

                if self.detect_outliers(col):

                    has_outliers = True

                    outlier_columns.append(col)

            except Exception:
                pass

        # =================================================
        # TEXT FEATURES
        # =================================================

        text_features = False

        text_columns = []

        for col in categorical_columns:

            try:

                avg_length = (

                    self.df[col]

                    .astype(str)

                    .str.len()

                    .mean()
                )

                if avg_length > 20:

                    text_features = True

                    text_columns.append(col)

            except Exception:
                pass

        # =================================================
        # FEATURE ANALYZER
        # =================================================

        feature_analyzer = FeatureAnalyzer(

            dataframe=self.df,

            target_column=self.target_column
        )

        # =================================================
        # IMPORTANT FEATURES
        # =================================================

        important_features = (
            feature_analyzer.best_features()
        )

        # =================================================
        # TOP FEATURES
        # =================================================

        top_features = (
            feature_analyzer.top_features()
        )

        # =================================================
        # COMBINED FEATURES
        # =================================================

        combined_features = (
            feature_analyzer.create_combined_features()
        )

        # =================================================
        # PROBLEM TYPE
        # =================================================

        problem_type = None

        if self.target_column is not None:

            try:

                target_values = self.df[
                    self.target_column
                ]

                unique_ratio = (

                    target_values.nunique() /

                    len(target_values)
                )

                if unique_ratio > 0.5:

                    problem_type = "Regression"

                else:

                    problem_type = "Classification"

            except Exception:

                problem_type = "Unknown"

        # =================================================
        # RETURN FACTS
        # =================================================

        return DatasetFact(

            rows=self.df.shape[0],

            columns=self.df.shape[1],

            numerical_columns=numerical_columns,

            categorical_columns=categorical_columns,

            missing_columns=missing_columns,

            skewed_columns=skewed_columns,

            outlier_columns=outlier_columns,

            text_columns=text_columns,

            important_features=important_features,

            top_features=top_features,

            combined_features=combined_features,

            target_column=self.target_column,

            problem_type=problem_type,

            has_missing_values=has_missing_values,

            numerical_features=(
                len(numerical_columns) > 0
            ),

            categorical_features=(
                len(categorical_columns) > 0
            ),

            high_dimensionality=(
                high_dimensionality
            ),

            has_outliers=has_outliers,

            imbalanced_dataset=False,

            text_features=text_features,

            skewed_data=(
                len(skewed_columns) > 0
            )
        )