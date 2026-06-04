# ============================================================
# ui/dashboard.py
# ============================================================

import streamlit as st
import pandas as pd


class Dashboard:

    def __init__(
        self,
        dataframe,
        dataset_fact,
        recommendations
    ):

        self.df = dataframe

        self.dataset_fact = dataset_fact

        self.recommendations = recommendations

    # ========================================================
    # MAIN RENDER
    # ========================================================

    def render(self):

        self.show_dataset_analysis()

        self.show_feature_importance()

        self.show_feature_combinations()

        self.show_generated_features()

        self.show_recommendations()

    # ========================================================
    # DATASET ANALYSIS
    # ========================================================

    def show_dataset_analysis(self):

        st.header("📊 Dataset Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Numerical Columns")

            numerical = self.dataset_fact.get(
                'numerical_columns',
                []
            )

            if numerical:

                for col in numerical:

                    st.write(f"• {col}")

        with col2:

            st.subheader("Categorical Columns")

            categorical = self.dataset_fact.get(
                'categorical_columns',
                []
            )

            if categorical:

                for col in categorical:

                    st.write(f"• {col}")

        st.subheader("Missing Value Columns")

        missing = self.dataset_fact.get(
            'missing_columns',
            []
        )

        if missing:

            for col in missing:

                st.write(f"• {col}")

        else:

            st.success(
                "No missing values detected."
            )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def show_feature_importance(self):

        features = self.dataset_fact.get(
            'important_features',
            []
        )

        if not features:

            return

        st.header("⭐ Best Features")

        feature_df = pd.DataFrame(features)

        st.dataframe(
            feature_df,
            use_container_width=True
        )

        st.info("""

These features were selected because they contain
the strongest predictive information related
to the target column.

Higher score = More important feature.

""")

    # ========================================================
    # FEATURE COMBINATIONS
    # ========================================================

    def show_feature_combinations(self):

        combinations = self.dataset_fact.get(
            'recommended_combinations',
            []
        )

        if not combinations:

            return

        st.header("🔗 Recommended Feature Combinations")

        for combo in combinations:

            with st.expander(

                f"{combo['feature_1']} + "
                f"{combo['feature_2']}"

            ):

                st.write(

                    f"### Combination Type\n"
                    f"{combo['combination_type']}"
                )

                st.write(

                    f"### Suggested Feature\n"
                    f"{combo['combination']}"
                )

                st.write(

                    f"### Why Combine?\n"
                    f"{combo['reason']}"
                )

    # ========================================================
    # GENERATED FEATURES
    # ========================================================

    def show_generated_features(self):

        features = self.dataset_fact.get(
            'combined_features',
            []
        )

        if not features:

            return

        st.header("🧠 Generated Features")

        for feature in features:

            with st.expander(
                feature['feature']
            ):

                st.write(

                    f"### Feature Type\n"
                    f"{feature['type']}"
                )

                st.write(

                    f"### Reason\n"
                    f"{feature['reason']}"
                )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def show_recommendations(self):

        st.header(
            "🚀 AI Recommendations"
        )

        if not self.recommendations:

            st.warning(
                "No recommendations generated."
            )

            return

        for rec in self.recommendations:

            with st.expander(
                rec['name']
            ):

                # ====================================================
                # CONFIDENCE
                # ====================================================

                st.write(

                    f"### Confidence\n"
                    f"{rec['confidence']}"
                )

                # ====================================================
                # REASON
                # ====================================================

                st.write(

                    f"### Reason\n"
                    f"{rec['reason']}"
                )

                # ====================================================
                # TECHNIQUES
                # ====================================================

                techniques = rec.get(
                    'techniques',
                    []
                )

                if techniques:

                    st.write(
                        "### Suggested Techniques"
                    )

                    for tech in techniques:

                        st.write(
                            f"• {tech}"
                        )

                # ====================================================
                # TARGET COLUMNS
                # ====================================================

                columns = rec.get(
                    'target_columns',
                    []
                )

                if columns:

                    st.write(
                        "### Target Columns"
                    )

                    for col in columns:

                        st.write(
                            f"• {col}"
                        )

                # ====================================================
                # EXTRA DETAILS
                # ====================================================

                extra = rec.get(
                    'extra_details',
                    []
                )

                if extra:

                    st.write(
                        "### Additional Details"
                    )

                    for detail in extra:

                        st.write(
                            f"• {detail}"
                        )