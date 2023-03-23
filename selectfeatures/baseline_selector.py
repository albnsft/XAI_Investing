import itertools
from abc import ABC

import pandas as pd


class BaseFeatureSelector(ABC):
    def __init__(self):
        self._fn_select = None

    def select_enumerate(self, all_columns, importance_df: pd.DataFrame):
        min_row = self._fn_select(importance_df)
        if min_row.empty:
            print("No removable features, selecting all columns")
            yield None, list(importance_df.index)
        else:
            print('Removable features according to selector: ')
            print(min_row.to_string())
            for k in range(1, len(min_row) + 1):
                combinations_tickers = list(itertools.combinations(min_row.index, k))
                for remove_columns in combinations_tickers:
                    if len(remove_columns) > 1: remove_columns = list(remove_columns)
                    row = min_row.loc[remove_columns].to_frame().T if len(remove_columns)==1 else min_row.loc[remove_columns]
                    columns = [col for col in all_columns if col not in remove_columns]
                    yield row, columns


class AllBellowZeroSelector(BaseFeatureSelector):
    def __init__(self):
        super().__init__()
        self._fn_select = lambda df: df[df['feature_importance'] <= 0].sort_values(
            by=['feature_importance'],
            ascending=[False])


class SelectKWorst(BaseFeatureSelector):
    def __init__(self, k):
        super().__init__()
        self._k = k
        self._fn_select = lambda df: df.sort_values(by=['feature_importance'], ascending=[False]).tail(self._k)
    

class FeatureSelectorBase(ABC):
    def __init__(self, k=1, selector: BaseFeatureSelector = None):
        self._columns = None
        self._k = k
        self._original_loss = None
        self.name = None
        self._selector = selector

    def transform(self, estimator, X_train, y_train, X_test, y_test):
        return self.fit_transform(estimator, X_train, y_train, X_test, y_test)

    def fit_transform(self, estimator, X_train, y_train, X_test, y_test):
        pass

    @property
    def baseline_loss(self):
        return self._original_loss

    def _feature_importance_permutation(self, estimator_fn, X_test, y_test):
        """Feature importance imputation via permutation importance"""
        pass