# ============================================================
# analyzer/feature_analyzer.py
# ============================================================

import pandas as pd
import numpy as np

from itertools import combinations

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


class FeatureAnalyzer:

    def __init__(
        self,
        dataframe,
        target_column=None
    ):

        self.df = dataframe.copy()

        self.target_column = target_column

    # ========================================================
    # ENCODE CATEGORICAL COLUMNS
    # ========================================================

    def encode_categorical(self, df):

        encoded_df = df.copy()

        # BOOLEAN → INTEGER
        encoded_df = encoded_df.replace({

            True: 1,
            False: 0
        })

        categorical_columns = encoded_df.select_dtypes(
            include=['object']
        ).columns

        for col in categorical_columns:

            try:

                encoder = LabelEncoder()

                encoded_df[col] = encoder.fit_transform(

                    encoded_df[col]
                    .astype(str)
                )

            except Exception:
                pass

        return encoded_df

    # ========================================================
    # DETECT ML PROBLEM TYPE
    # ========================================================

    def detect_problem_type(self, y):

        unique_ratio = (
            y.nunique() / len(y)
        )

        if unique_ratio > 0.5:

            return "Regression"

        return "Classification"

    # ========================================================
    # PREPARE DATA
    # ========================================================

    def prepare_data(self):

        if self.target_column is None:

            return None, None, None

        encoded_df = self.encode_categorical(
            self.df
        )

        X = encoded_df.drop(
            columns=[self.target_column]
        )

        y = encoded_df[self.target_column]

        X = X.fillna(0)

        y = y.fillna(0)

        problem_type = self.detect_problem_type(
            y
        )

        return X, y, problem_type

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def feature_importance(self):

        if self.target_column is None:

            return []

        try:

            X, y, problem_type = self.prepare_data()

            # =================================================
            # MUTUAL INFORMATION
            # =================================================

            if problem_type == "Regression":

                mi_scores = mutual_info_regression(
                    X,
                    y
                )

                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )

            else:

                mi_scores = mutual_info_classif(
                    X,
                    y
                )

                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )

            # =================================================
            # RANDOM FOREST IMPORTANCE
            # =================================================

            model.fit(X, y)

            rf_scores = model.feature_importances_

            # =================================================
            # COMBINE SCORES
            # =================================================

            results = []

            for col, mi, rf in zip(
                X.columns,
                mi_scores,
                rf_scores
            ):

                final_score = (
                    float(mi) * 0.5 +
                    float(rf) * 0.5
                )

                # =============================================
                # IMPORTANCE LEVEL
                # =============================================

                if final_score >= 0.7:

                    strength = "VERY STRONG"

                elif final_score >= 0.4:

                    strength = "STRONG"

                elif final_score >= 0.2:

                    strength = "MEDIUM"

                else:

                    strength = "WEAK"

                # =============================================
                # SMART EXPLANATION
                # =============================================

                reason = self.generate_feature_reason(
                    col,
                    final_score,
                    strength
                )

                results.append({

                    "column": col,

                    "score": round(
                        final_score,
                        4
                    ),

                    "mutual_information":
                        round(float(mi), 4),

                    "random_forest_importance":
                        round(float(rf), 4),

                    "strength":
                        strength,

                    "reason":
                        reason
                })

            # =================================================
            # SORT
            # =================================================

            results = sorted(

                results,

                key=lambda x: x['score'],

                reverse=True
            )

            return results

        except Exception as e:

            print(
                f"[ERROR] Feature importance failed: {e}"
            )

            return []

    # ========================================================
    # GENERATE FEATURE REASON
    # ========================================================

    def generate_feature_reason(
        self,
        column,
        score,
        strength
    ):

        if strength == "VERY STRONG":

            return (
                f"{column} has a very strong "
                f"influence on predicting "
                f"{self.target_column}."
            )

        elif strength == "STRONG":

            return (
                f"{column} contains important "
                f"predictive information."
            )

        elif strength == "MEDIUM":

            return (
                f"{column} moderately affects "
                f"prediction performance."
            )

        return (
            f"{column} has weak predictive "
            f"impact but may still help "
            f"in combination with other features."
        )

    # ========================================================
    # BEST FEATURES ONLY
    # ========================================================

    def best_features(
        self,
        top_n=5
    ):

        features = self.feature_importance()

        return features[:top_n]

    # ========================================================
    # TOP FEATURES
    # ========================================================

    def top_features(
        self,
        top_n=5
    ):

        return self.best_features(top_n)

    # ========================================================
    # DETECT COMBINATION TYPE
    # ========================================================

    def detect_combination_type(
        self,
        col1,
        col2
    ):

        numeric_columns = list(

            self.df.select_dtypes(

                include=['int64', 'float64']

            ).columns
        )

        if (

            col1 in numeric_columns
            and
            col2 in numeric_columns
        ):

            return "Mathematical Combination"

        elif (

            (col1 in numeric_columns)

            !=

            (col2 in numeric_columns)
        ):

            return "Grouped Statistical Combination"

        return "Categorical/Text Combination"

    # ========================================================
    # RECOMMEND COMBINATIONS
    # ========================================================

    def recommend_feature_combinations(
        self,
        top_n=5
    ):

        recommendations = []

        top_features = self.top_features(
            top_n=top_n
        )

        top_columns = [

            feature['column']

            for feature in top_features
        ]

        for col1, col2 in combinations(
            top_columns,
            2
        ):

            if col1 == self.target_column:
                continue

            if col2 == self.target_column:
                continue

            combination_type = (
                self.detect_combination_type(
                    col1,
                    col2
                )
            )

            reason = (

                f"Combining {col1} and {col2} "
                f"may improve model accuracy "
                f"because both contain strong "
                f"predictive information."
            )

            recommendations.append({

                "feature_1": col1,

                "feature_2": col2,

                "combination":
                    f"{col1}_{col2}",

                "combination_type":
                    combination_type,

                "reason":
                    reason
            })

        return recommendations

    # ========================================================
    # CREATE COMBINED FEATURES
    # ========================================================

    def create_combined_features(self):

        combined = []

        numeric_columns = list(

            self.df.select_dtypes(

                include=['int64', 'float64']

            ).columns
        )

        if len(numeric_columns) >= 2:

            for col1, col2 in combinations(
                numeric_columns,
                2
            ):

                # SUM
                sum_feature = (
                    f"{col1}_{col2}_sum"
                )

                self.df[sum_feature] = (

                    self.df[col1].fillna(0) +

                    self.df[col2].fillna(0)
                )

                combined.append({

                    "feature":
                        sum_feature,

                    "type":
                        "SUM",

                    "reason":
                        f"Combining {col1} "
                        f"and {col2} may "
                        f"capture hidden "
                        f"numerical patterns."
                })

                # RATIO
                ratio_feature = (
                    f"{col1}_{col2}_ratio"
                )

                self.df[ratio_feature] = (

                    self.df[col1].fillna(1) /

                    (
                        self.df[col2].fillna(1)
                        + 1e-5
                    )
                )

                combined.append({

                    "feature":
                        ratio_feature,

                    "type":
                        "RATIO",

                    "reason":
                        f"Ratio between "
                        f"{col1} and {col2} "
                        f"may improve "
                        f"prediction quality."
                })

        return combined

    # ========================================================
    # FULL REPORT
    # ========================================================

    def full_report(self):

        report = {

            "target_column":
                self.target_column,

            "feature_importance":
                self.feature_importance(),

            "best_features":
                self.best_features(),

            "top_features":
                self.top_features(),

            "combined_features":
                self.create_combined_features(),

            "recommended_combinations":
                self.recommend_feature_combinations()
        }

        return report