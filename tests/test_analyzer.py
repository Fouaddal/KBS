from analyzer.dataset_analyzer import DatasetAnalyzer
import pandas as pd


def test_analyzer():

    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })

    analyzer = DatasetAnalyzer(df)

    result = analyzer.analyze()

    assert result["numerical_features"] is True