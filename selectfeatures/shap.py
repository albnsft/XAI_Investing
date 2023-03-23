from selectfeatures.baseline_selector import *
import shap
import numpy as np


class SHAPFeatureImportanceSelector(FeatureSelectorBase):

    def __init__(self, k=5, num_rounds=30, seed=42):
        selector = SelectKWorst(k=k)
        super(SHAPFeatureImportanceSelector, self).__init__(k=k, selector=selector)
        self._seed = seed
        self._num_rounds = num_rounds
        self.metric = 'accuracy'

    def fit_transform(self, clf, X_train, y_train, X_test, y_test):
        # ####***************LIME feature importance***********************
        print('*' * 20, 'SHAP Feature Importance', '*' * 20)
        estimator = clf.model
        permutation_importance_s = self.__compute_shape_importance(X_train, y_train, X_test, y_test, estimator)
        print(permutation_importance_s.sort_values('feature_importance', ascending=False).to_string())
        for min_row, columns in self._selector.select_enumerate(X_test.columns, permutation_importance_s):
            yield min_row, columns

    def __compute_shape_importance(self, X_train, y_train, X_test, y_test, estimator):
        columns = list(X_train.columns)
        X_test = estimator['scaler'].transform(X_test.values)
        tree_explainer = shap.TreeExplainer(estimator['clf'])
        shap_values = tree_explainer.shap_values(X_test)
        res = pd.DataFrame(shap_values[0], columns=columns)
        res = res.abs()
        imp_ci = 1.96 * res.std(ddof=0) / np.sqrt(len(res))
        permutation_importance = pd.DataFrame(
            dict(
                feature_importance=res.mean(),
                ci_fixed=imp_ci,
            ),
        )
        return permutation_importance


