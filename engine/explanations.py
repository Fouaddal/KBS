# ============================================================
# engine/explanations.py
# ============================================================

class ExplanationSystem:

    explanations = {

        # ====================================================
        # NORMALIZATION
        # ====================================================

        "normalization": {

            "title":
                "Normalization / Standardization",

            "description":
                "Normalization scales numerical values "
                "to a similar range so machine learning "
                "algorithms can learn effectively.",

            "why":
                "Different numeric scales can negatively "
                "impact ML model performance.",

            "benefits": [

                "Improves convergence speed",

                "Prevents feature dominance",

                "Improves model accuracy"
            ],

            "techniques": [

                "MinMaxScaler",

                "StandardScaler"
            ]
        },

        # ====================================================
        # ENCODING
        # ====================================================

        "encoding": {

            "title":
                "Categorical Encoding",

            "description":
                "Converts categorical text values "
                "into numerical representations.",

            "why":
                "Machine learning algorithms cannot "
                "understand raw text categories.",

            "benefits": [

                "Makes categorical data usable",

                "Improves ML compatibility",

                "Enables statistical learning"
            ],

            "techniques": [

                "Label Encoding",

                "One-Hot Encoding"
            ]
        },

        # ====================================================
        # MISSING VALUES
        # ====================================================

        "missing_values": {

            "title":
                "Missing Value Imputation",

            "description":
                "Handles empty or null values in dataset.",

            "why":
                "Missing values may reduce model quality "
                "and create training instability.",

            "benefits": [

                "Improves dataset completeness",

                "Prevents model errors",

                "Improves prediction quality"
            ],

            "techniques": [

                "Mean Imputation",

                "Median Imputation",

                "Most Frequent Imputation"
            ]
        },

        # ====================================================
        # OUTLIERS
        # ====================================================

        "outliers": {

            "title":
                "Outlier Detection & Removal",

            "description":
                "Detects abnormal values far away "
                "from normal distribution.",

            "why":
                "Outliers can distort training patterns "
                "and reduce model accuracy.",

            "benefits": [

                "Improves data quality",

                "Reduces statistical distortion",

                "Improves model robustness"
            ],

            "techniques": [

                "IQR Method",

                "Z-Score"
            ]
        },

        # ====================================================
        # FEATURE SELECTION
        # ====================================================

        "feature_selection": {

            "title":
                "Feature Selection",

            "description":
                "Selects the most useful columns "
                "for machine learning.",

            "why":
                "Too many features may cause overfitting "
                "and increase model complexity.",

            "benefits": [

                "Reduces overfitting",

                "Improves training speed",

                "Improves model interpretability"
            ],

            "techniques": [

                "SelectKBest",

                "Chi-Square",

                "Mutual Information"
            ]
        },

        # ====================================================
        # NLP
        # ====================================================

        "nlp": {

            "title":
                "NLP Preprocessing",

            "description":
                "Processes textual data for NLP models.",

            "why":
                "Text data requires preprocessing "
                "before machine learning usage.",

            "benefits": [

                "Improves text understanding",

                "Extracts semantic meaning",

                "Converts text to vectors"
            ],

            "techniques": [

                "Tokenization",

                "TF-IDF",

                "Stopword Removal"
            ]
        },

        # ====================================================
        # SKEWED DATA
        # ====================================================

        "skewed_data": {

            "title":
                "Log Transformation",

            "description":
                "Transforms skewed distributions "
                "into more normal distributions.",

            "why":
                "Highly skewed data may violate "
                "ML statistical assumptions.",

            "benefits": [

                "Improves normality",

                "Reduces variance",

                "Improves model performance"
            ],

            "techniques": [

                "Log Transform",

                "Box-Cox Transform"
            ]
        }
    }

    # ========================================================
    # GET FULL EXPLANATION
    # ========================================================

    @staticmethod
    def explain(topic):

        return ExplanationSystem.explanations.get(

            topic,

            {

                "title": "Unknown",

                "description":
                    "No description available.",

                "why":
                    "No explanation available.",

                "benefits": [],

                "techniques": []
            }
        )