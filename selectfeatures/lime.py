import lime
import lime.lime_tabular
import numpy as np
from lime import submodular_pick

from selectfeatures.baseline_selector import *


class LIMEFeatureImportanceSelector(FeatureSelectorBase):

    def __init__(self, k=5, num_rounds=30, seed=42):
        selector = SelectKWorst(k=k)
        super(LIMEFeatureImportanceSelector, self).__init__(k=k, selector=selector)
        self._seed = seed
        self._num_rounds = num_rounds
        self.metric = 'accuracy'

    def fit_transform(self, clf, X_train, y_train, X_test, y_test):
        # ####***************LIME feature importance***********************
        print('*' * 20, 'LIME Feature Importance', '*' * 20)
        estimator = clf.model
        permutation_importance_s = self.__compute_lime_importance(X_train, y_train, X_test, y_test, estimator)
        print(permutation_importance_s.sort_values('feature_importance', ascending=False).to_string())
        for min_row, columns in self._selector.select_enumerate(X_test.columns, permutation_importance_s):
            yield min_row, columns

    def __compute_lime_importance(self, X_train, y_train, X_test, y_test, estimator):
        columns = list(X_train.columns)
        X_train = estimator['scaler'].fit_transform(X_train.values)
        X_test = estimator['scaler'].transform(X_test.values)
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train,
                                                                training_labels=y_train.values,
                                                                feature_names=columns,
                                                                verbose=False, mode='classification'
                                                                , discretize_continuous=False
                                                                , random_state=self._seed,
                                                                class_names=['short', 'long'])
        sp_obj = submodular_pick.SubmodularPick(lime_explainer, X_test, estimator['clf'].predict_proba,
                                                num_features=len(columns),
                                                num_exps_desired=self._num_rounds)
        res = pd.DataFrame([dict(this.as_list(label=this.available_labels()[0])) for this in sp_obj.explanations],
                           columns=columns)
        res = res.abs()
        imp_ci = 1.96 * res.std(ddof=0) / np.sqrt(len(res))
        permutation_importance = pd.DataFrame(
            dict(
                feature_importance=res.mean(),
                ci_fixed=imp_ci,
            ),
        )
        """
        res['prediction'] = [this.available_labels()[0] for this in sp_obj.explanations]
        grped_coeff = res.groupby("prediction").mean().T
        grped_coeff["abs"] = np.abs(grped_coeff.iloc[:, 0])
        grped_coeff.sort_values("abs", inplace=True, ascending=False)
        grped_coeff.head(25).sort_values("abs", ascending=True).drop("abs", axis=1).plot(kind="barh")
        """
        return permutation_importance
