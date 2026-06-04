from preprocessing.normalization import NormalizationProcessor
import numpy as np


def test_normalization():

    data = np.array([[1], [2], [3]])

    scaled = NormalizationProcessor.minmax(data)

    assert scaled.max() <= 1