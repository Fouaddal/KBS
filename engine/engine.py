# ============================================================
# engine/engine.py
# ============================================================

from engine.rules import (
    FeatureEngineeringExpertSystem
)


# ============================================================
# PRINT PROFESSIONAL REPORT
# ============================================================

def print_recommendations(recommendations):

    print("\n")

    print("=" * 70)

    print(
        "FEATURE ENGINEERING EXPERT SYSTEM REPORT"
    )

    print("=" * 70)

    for index, rec in enumerate(
        recommendations,
        start=1
    ):

        print(f"\n[{index}] {rec['name']}")

        print("-" * 50)

        print(
            f"Confidence : "
            f"{rec['confidence']}"
        )

        print(
            f"Reason     : "
            f"{rec['reason']}"
        )

        # ====================================================
        # DESCRIPTION
        # ====================================================

        if rec.get("description"):

            print(
                f"\nDescription"
            )

            print(
                f"  {rec['description']}"
            )

        # ====================================================
        # WHY
        # ====================================================

        if rec.get("why"):

            print(
                f"\nWhy?"
            )

            print(
                f"  {rec['why']}"
            )

        # ====================================================
        # BENEFITS
        # ====================================================

        if rec.get("benefits"):

            print(
                "\nBenefits"
            )

            for benefit in rec[
                "benefits"
            ]:

                print(
                    f"  • {benefit}"
                )

        # ====================================================
        # TECHNIQUES
        # ====================================================

        if rec.get("techniques"):

            print(
                "\nSuggested Techniques"
            )

            for technique in rec[
                "techniques"
            ]:

                print(
                    f"  • {technique}"
                )

        # ====================================================
        # TARGET COLUMNS
        # ====================================================

        if rec.get("target_columns"):

            print(
                "\nTarget Columns"
            )

            for column in rec[
                "target_columns"
            ]:

                print(
                    f"  • {column}"
                )

        print("\n" + "=" * 70)


# ============================================================
# RUN ENGINE
# ============================================================

def run_engine(dataset_information):

    engine = (
        FeatureEngineeringExpertSystem()
    )

    engine.reset()

    engine.declare(dataset_information)

    engine.run()

    print_recommendations(
        engine.recommendations
    )

    return engine.recommendations