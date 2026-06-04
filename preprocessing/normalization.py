from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class NormalizationProcessor:

    @staticmethod
    def minmax(data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)

    @staticmethod
    def standard(data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)