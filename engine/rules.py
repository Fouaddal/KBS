# ============================================================
# engine/rules.py
# ============================================================

from experta import *

from engine.facts import DatasetFact

from engine.explanations import (
    ExplanationSystem
)

from engine.confidence import (
    ConfidenceCalculator
)


class FeatureEngineeringExpertSystem(
    KnowledgeEngine
):

    def __init__(self):

        super().__init__()

        self.recommendations = []

    # ========================================================
    # ADD RECOMMENDATION
    # ========================================================

    def add_recommendation(

        self,

        topic,

        confidence_score,

        target_columns=None
    ):

        explanation = (
            ExplanationSystem.explain(
                topic
            )
        )

        recommendation = {

            "name":
                explanation["title"],

            "reason":
                explanation["why"],

            "description":
                explanation["description"],

            "benefits":
                explanation["benefits"],

            "confidence":
                ConfidenceCalculator.calculate(
                    confidence_score
                ),

            "techniques":
                explanation["techniques"],

            "target_columns":
                target_columns or []
        }

        self.recommendations.append(
            recommendation
        )

        print(
            f"[RECOMMENDATION] "
            f"{recommendation['name']}"
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    @Rule(
        DatasetFact(
            has_missing_values=True,
            missing_columns=MATCH.columns
        )
    )
    def missing_values_rule(
        self,
        columns
    ):

        self.add_recommendation(

            topic="missing_values",

            confidence_score=0.95,

            target_columns=columns
        )

    # ========================================================
    # NUMERICAL FEATURES
    # ========================================================

    @Rule(
        DatasetFact(
            numerical_features=True,
            numerical_columns=MATCH.columns
        )
    )
    def normalization_rule(
        self,
        columns
    ):

        self.add_recommendation(

            topic="normalization",

            confidence_score=0.80,

            target_columns=columns
        )

    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    @Rule(
        DatasetFact(
            categorical_features=True,
            categorical_columns=MATCH.columns
        )
    )
    def encoding_rule(
        self,
        columns
    ):

        self.add_recommendation(

            topic="encoding",

            confidence_score=0.90,

            target_columns=columns
        )

    # ========================================================
    # OUTLIERS
    # ========================================================

    @Rule(
        DatasetFact(
            has_outliers=True,
            outlier_columns=MATCH.columns
        )
    )
    def outliers_rule(
        self,
        columns
    ):

        self.add_recommendation(

            topic="outliers",

            confidence_score=0.80,

            target_columns=columns
        )

    # ========================================================
    # HIGH DIMENSIONALITY
    # ========================================================

    @Rule(
        DatasetFact(
            high_dimensionality=True
        )
    )
    def feature_selection_rule(
        self
    ):

        self.add_recommendation(

            topic="feature_selection",

            confidence_score=0.85
        )

    # ========================================================
    # TEXT FEATURES
    # ========================================================

    @Rule(
        DatasetFact(
            text_features=True,
            text_columns=MATCH.columns
        )
    )
    def nlp_rule(
        self,
        columns
    ):

        self.add_recommendation(

            topic="nlp",

            confidence_score=0.90,

            target_columns=columns
        )

    # ========================================================
    # SKEWED DATA
    # ========================================================

    @Rule(
        DatasetFact(
            skewed_data=True,
            skewed_columns=MATCH.columns
        )
    )
    def skewed_rule(
        self,
        columns
    ):

        self.add_recommendation(

            topic="skewed_data",

            confidence_score=0.80,

            target_columns=columns
        )