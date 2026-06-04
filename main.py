import pandas as pd

from analyzer.dataset_analyzer import (
    DatasetAnalyzer
)

from engine.engine import run_engine


def main():

    df = pd.read_csv(
        r"C:\Users\Fouad\Downloads\3csv\JOR\synonyms.csv"
    )

    analyzer = DatasetAnalyzer(df)

    dataset_fact = analyzer.analyze()

    recommendations = run_engine(
        dataset_fact
    )

    print("\n")
    print("=" * 65)
    print("FEATURE ENGINEERING EXPERT SYSTEM REPORT")
    print("=" * 65)

    print(f"\nRows    : {dataset_fact['rows']}")
    print(f"Columns : {dataset_fact['columns']}")

    print("\nNUMERICAL COLUMNS")
    print("-" * 40)

    for col in dataset_fact["numerical_columns"]:
        print(f"• {col}")

    print("\nCATEGORICAL COLUMNS")
    print("-" * 40)

    for col in dataset_fact["categorical_columns"]:
        print(f"• {col}")

    print("\nMISSING VALUE COLUMNS")
    print("-" * 40)

    if dataset_fact["missing_columns"]:

        for col in dataset_fact["missing_columns"]:
            print(f"• {col}")

    else:
        print("No missing columns")

    print("\n")
    print("=" * 65)
    print("RECOMMENDATIONS")
    print("=" * 65)

    for idx, rec in enumerate(
        recommendations,
        start=1
    ):

        print(f"\n[{idx}] {rec['name']}")
        print("-" * 50)

        print(
            f"Confidence : "
            f"{rec['confidence']}"
        )

        print(
            f"Reason     : "
            f"{rec['reason']}"
        )

        print("\nSuggested Techniques")

        for tech in rec["techniques"]:
            print(f"  • {tech}")

        if rec["target_columns"]:

            print("\nTarget Columns")

            for col in rec["target_columns"]:
                print(f"  • {col}")

        print("\n" + "=" * 65)

    print("\nSYSTEM SUMMARY")
    print("-" * 40)

    print(
        f"Total Recommendations : "
        f"{len(recommendations)}"
    )

    print("Engine Status         : SUCCESS")

    print("=" * 65)


if __name__ == "__main__":
    main()