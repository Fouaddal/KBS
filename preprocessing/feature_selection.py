from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


class FeatureSelectionProcessor:

    @staticmethod
    def select_best_features(X, y, k=5):
        selector = SelectKBest(score_func=chi2, k=k)
        return selector.fit_transform(X, y)