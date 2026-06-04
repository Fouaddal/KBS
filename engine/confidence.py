# ============================================================
# engine/confidence.py
# ============================================================

class ConfidenceCalculator:

    @staticmethod
    def calculate(score):

        if score >= 0.90:
            return "VERY HIGH"

        elif score >= 0.75:
            return "HIGH"

        elif score >= 0.50:
            return "MEDIUM"

        return "LOW"