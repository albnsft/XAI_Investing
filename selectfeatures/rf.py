from selectfeatures.baseline_selector import *
import numpy as np


class RFFeatureImportanceSelector(FeatureSelectorBase):
    def __init__(self, k=5):
        selector = SelectKWorst(k=k)
        super(RFFeatureImportanceSelector, self).__init__(k=k, selector=selector)

    def fit_transform(self, clf, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_test: pd.DataFrame):
        print('*' * 20, 'Impurity Feature importance', '*' * 20)
        estimator = clf.model
        permutation_importance_s = self.__compute_feature_importance(X_test, y_test, estimator)
        print(permutation_importance_s.sort_values('feature_importance', ascending=False).to_string())
        for min_row, columns in self._selector.select_enumerate(X_test.columns, permutation_importance_s):
            yield min_row, columns

    def __compute_feature_importance(self, X_test, y_test, estimator):
        all_importance_values = [tree.feature_importances_ for i, tree in enumerate(estimator['clf'].estimators_)]
        res = pd.DataFrame(data=all_importance_values, columns=X_test.columns,
                           index=range(0, len(estimator['clf'].estimators_)))

        imp_ci = 1.96 * res.std(ddof=0) / np.sqrt(len(estimator['clf'].estimators_))
        permutation_importance = pd.DataFrame(
            dict(
                feature_importance=res.mean(),
                ci_fixed=imp_ci,
            ),
        )

        return permutation_importance