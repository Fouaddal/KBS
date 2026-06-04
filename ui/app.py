# ============================================================
# ui/app.py
# ============================================================

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

import streamlit as st
import pandas as pd

from analyzer.dataset_analyzer import DatasetAnalyzer
from engine.engine import run_engine
from ui.dashboard import Dashboard


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="Feature Engineering Expert System",

    page_icon="🧠",

    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""

<style>

.main {
    background-color: #0E1117;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
}

</style>

""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.title(
    "🧠 Feature Engineering Expert System"
)

st.markdown("""

This intelligent Knowledge-Based System analyzes datasets
and recommends the best Feature Engineering strategies.

### System Capabilities

- Detect missing values
- Detect outliers
- Detect skewed distributions
- Detect text features
- Rank important features
- Recommend feature combinations
- Generate engineered features
- Explain all recommendations

""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ AI Expert System")

st.sidebar.success("System Status: ACTIVE")

st.sidebar.markdown("""

### Technologies

- Experta
- Pandas
- Scikit-learn
- Streamlit
- Rule-Based AI
- Explainable AI

""")

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(

    "📂 Upload CSV Dataset",

    type=["csv"]
)

# ============================================================
# MAIN SYSTEM
# ============================================================

if uploaded_file:

    # ========================================================
    # LOAD DATASET
    # ========================================================

    try:

        df = pd.read_csv(uploaded_file)

        st.success(
            "✅ Dataset Loaded Successfully"
        )

    except Exception as e:

        st.error(
            f"Dataset loading failed: {e}"
        )

        st.stop()

    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.header("📊 Dataset Preview")

    st.dataframe(df.head())

    # ========================================================
    # METRICS
    # ========================================================

    st.header("📈 Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:

        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    # ========================================================
    # TARGET COLUMN
    # ========================================================

    st.header("🎯 Target Selection")

    target_column = st.selectbox(

        "Select Target Column",

        options=df.columns
    )

    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    with st.spinner(
        "Analyzing dataset..."
    ):

        analyzer = DatasetAnalyzer(

            dataframe=df,

            target_column=target_column
        )

        dataset_fact = analyzer.analyze()

        recommendations = run_engine(
            dataset_fact
        )

    st.success(
        "✅ Analysis Completed"
    )

    # ========================================================
    # DASHBOARD
    # ========================================================

    dashboard = Dashboard(

        dataframe=df,

        dataset_fact=dataset_fact,

        recommendations=recommendations
    )

    dashboard.render()