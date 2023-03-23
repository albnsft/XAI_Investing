import numpy as np
import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

from selectfeatures.baseline_selector import *


class PISelectorBase(FeatureSelectorBase):
    def __init__(self, k=0,  num_rounds=30, seed=42):
        selector = AllBellowZeroSelector()
        super().__init__(k=k, selector=selector)
        self._seed = seed
        self._num_rounds = num_rounds
        self.metric = 'accuracy'

    def _loss(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def fit_transform(self, clf, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
        print('*' * 20, 'Permutation Feature Importance', '*' * 20)
        self._original_loss = clf.report_val[self.metric].values[0]
        estimator = clf.model
        permutation_importance_s = self.__compute_permutation_importance(X_test, y_test, estimator)
        print(permutation_importance_s.sort_values('feature_importance', ascending=False).to_string())

        for min_row, columns in self._selector.select_enumerate(X_test.columns, permutation_importance_s):
            yield min_row, columns

    def __compute_permutation_importance(self, X_test, y_test, estimator):
        res, res_error = self._feature_importance_permutation(estimator_fn=estimator.predict, X_test=X_test,
                                                              y_test=y_test['label'])

        count_check = res_error.applymap(lambda x: 1 if x <= self._original_loss else 0)
        imp_ci = 1.96 * res.std(ddof=0) / np.sqrt(self._num_rounds)
        permutation_importance = pd.DataFrame(
            dict(
                feature_importance=res.mean(),
                ci_fixed=imp_ci,
            ),
        )

        return permutation_importance


class PISelector(PISelectorBase):

    def __init__(self, k=0, num_rounds=50, seed=42):
        super().__init__(k, num_rounds, seed)
        self.name = "pi2_{0}".format("all" if k == 0 else k)

    def _feature_importance_permutation(self, estimator_fn, X_test, y_test):

        rng = np.random.RandomState(self._seed)
        X_test_v = X_test.values
        y_test_v = y_test.values
        mean_importance_values = np.zeros(X_test_v.shape[1])
        all_importance_values = np.zeros((X_test_v.shape[1], self._num_rounds))
        all_error_values = np.zeros((X_test_v.shape[1], self._num_rounds))

        for round_idx in range(self._num_rounds):
            for col_idx in range(X_test_v.shape[1]):
                save_col = X_test_v[:, col_idx].copy()
                rng.shuffle(X_test_v[:, col_idx])
                new_score = self._loss(y_test_v, estimator_fn(X_test_v))
                X_test_v[:, col_idx] = save_col
                importance = (self._original_loss - new_score) / self._original_loss
                mean_importance_values[col_idx] += importance
                all_importance_values[col_idx, round_idx] = importance
                all_error_values[col_idx, round_idx] = new_score

        all_feat_imp_df = pd.DataFrame(data=np.transpose(all_importance_values), columns=X_test.columns,
                                       index=range(0, self._num_rounds))
        all_trial_errors_df = pd.DataFrame(data=np.transpose(all_error_values), columns=X_test.columns,
                                           index=range(0, self._num_rounds))
        return all_feat_imp_df, all_trial_errors_df
