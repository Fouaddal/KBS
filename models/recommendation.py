class RecommendationModel:

    def __init__(self, name, reason, confidence):
        self.name = name
        self.reason = reason
        self.confidence = confidence

    def to_dict(self):

        return {
            "name": self.name,
            "reason": self.reason,
            "confidence": self.confidence
        }