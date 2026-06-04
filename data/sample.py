import pandas as pd


def load_sample():

    data = {
        "age": [25, 30, None, 40, 22],
        "salary": [50000, 60000, 70000, None, 52000],
        "gender": ["M", "F", "F", "M", "M"],
        "target": [1, 0, 1, 0, 1]
    }

    return pd.DataFrame(data)