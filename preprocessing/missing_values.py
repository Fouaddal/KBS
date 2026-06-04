from sklearn.impute import SimpleImputer


class MissingValueProcessor:

    @staticmethod
    def mean_imputation(data):
        imputer = SimpleImputer(strategy='mean')
        return imputer.fit_transform(data)

    @staticmethod
    def most_frequent_imputation(data):
        imputer = SimpleImputer(strategy='most_frequent')
        return imputer.fit_transform(data)