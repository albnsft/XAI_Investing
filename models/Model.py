from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, brier_score_loss, roc_auc_score, balanced_accuracy_score

from env.FinancialEnvironment import FinancialEnvironment
import pickle
import os
from pylab import plt, mpl
from env.utils import plot_final
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 1000
mpl.rcParams['font.family'] = 'serif'


class Classifier:
    def __init__(
            self,
            learn_env: FinancialEnvironment = None,
            valid_env: FinancialEnvironment = None,
            test_env: FinancialEnvironment = None,
            model_type: str = None,
            method_type: str = None,
            ticker: str = None,
            columns: list = None,
            removed_columns: list = None,
            verbose: bool = False,
            seed: int = 42,
            n_iter_opt: int = 100,
            n_cv: int = 5,
    ):
        self.learn_env = learn_env
        self.valid_env = valid_env
        self.test_env = test_env
        self.model_type = model_type
        self.method_type = method_type
        self.ticker = ticker
        self.columns = columns
        self.removed_columns = removed_columns
        self.verbose = verbose
        self.seed = seed
        self.n_iter_opt = n_iter_opt
        self.n_cv = n_cv
        self.best_params = {'rf': {}, 'lgb': {}, 'xgb': {}}
        self.init()

    def init(self):
        if self.model_type not in self.best_params.keys():
            raise Exception(f'{self.model_type} not been implemented')

        base = os.path.dirname(os.path.abspath(__file__))
        path_params = os.path.join(base, 'savings', self.ticker)
        self.path_params = os.path.join(path_params, f'{self.model_type}_params.pkl')
        self.path_results = os.path.join(base.replace('\models', ''), 'results', self.ticker)
        if os.path.exists(self.path_params):
            with open(self.path_params, "rb") as file:
                self.best_params[self.model_type] = pickle.load(file)
        if not os.path.exists(path_params): os.makedirs(path_params)
        if not os.path.exists(self.path_results): os.makedirs(self.path_results)

        if self.columns is not None:
            self.X_train = self.learn_env.X[self.columns].copy()
            self.X_val = self.valid_env.X[self.columns].copy()
            if self.test_env: self.X_test = self.test_env.X[self.columns].copy()
        else:
            self.X_train = self.learn_env.X.copy()
            self.X_val= self.valid_env.X.copy()
            if self.test_env: self.X_test= self.test_env.X.copy()
        self.y_train = self.learn_env.y.copy()
        self.y_val = self.valid_env.y.copy()
        if self.test_env: self.y_test = self.test_env.y.copy()

    def learn(self, test_info: bool=False):
        if not self.best_params[self.model_type]:
            cv = TimeSeriesSplit(n_splits=self.n_cv)
            self.model = GridSearchCV(self.models[self.model_type], self.space_params[self.model_type], cv=cv,
                                            scoring='accuracy',
                                            return_train_score=True, n_jobs=-1, verbose=1) #, n_iter=self.n_iter_opt, random_state=self.seed)
            self.model.fit(self.X_train.values, self.y_train.values.ravel())
            self.print_metrics_grid(self.model, self.n_cv)
            self.best_params[self.model_type] = {k.replace('clf__', ''): v for k, v in self.model.best_params_.items()}
            with open(self.path_params, "wb") as file:
                pickle.dump(self.best_params[self.model_type], file)
            self.model = self.model.best_estimator_
        else:
            X, y = pd.concat([self.X_train, self.X_val]), pd.concat([self.y_train, self.y_val])
            cv = TimeSeriesSplit(test_size=len(self.X_val), n_splits=self.n_cv)
            self.model = self.models[self.model_type]
            self.score = cross_validate(self.model, X.values, y.values.ravel(),
                                        scoring=['accuracy'], n_jobs=-1, verbose=0, cv=cv)
            if self.verbose: self.print_metrics_crossval(self.score, self.n_cv)
            self.model.fit(self.X_train.values, self.y_train.values.ravel())

        self.y_val, self.report_val, self.confusion_val, self.strat_val = self.report(self.X_val, self.y_val, self.valid_env)
        if self.test_env is not None:
            self.y_test, self.report_test, self.confusion_test, self.strat_test = self.report(self.X_test, self.y_test, self.test_env)
        if self.verbose:
            plot_final(self.ticker, self.model_type, self.method_type, self.report_val, self.confusion_val, self.strat_val,
                       self.report_test, self.confusion_test, self.strat_test, self.path_results, self.removed_columns)

        if test_info:
            return self.report_test['outperformance'].values[0]

    def report(self, X: pd.DataFrame, y: pd.DataFrame, env: FinancialEnvironment):
        type_env = 'Validation' if env == self.valid_env else 'Testing'
        y['proba'] = self.model.predict_proba(X.values)[:, 1]
        y['predicted'] = np.where(y['proba'] > 0.5, 1, 0)
        #classification metrics
        rept_base = classification_report(y['label'], y['predicted'], target_names=['short', 'long'], output_dict=True, zero_division=1)
        rept = pd.DataFrame.from_dict(rept_base['weighted avg'], orient='index').T
        rept['accuracy'] = rept_base['accuracy']
        rept['brier'] = brier_score_loss(y['label'], y['proba'])
        rept['auc'] = roc_auc_score(y['label'], y['proba'])
        confusion = confusion_matrix(y['label'], y['predicted'])
        cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['short', 'long'])
        #financial metrics
        strat, operf = env.backtest(y['proba'], type_env=type_env, type_algo=self.model_type, verbose=self.verbose)
        rept['outperformance'] = operf
        return y, rept, cm_display, strat

    @staticmethod
    def print_metrics_grid(grid_search, n_cv):
        cvres = grid_search.cv_results_
        hyperparams = list(cvres["params"][0].keys())
        results = pd.DataFrame(index=range(len(cvres["mean_test_score"])),
                               columns=['mean_accuracy', 'std_accuracy'] + hyperparams)
        results['mean_accuracy'] = cvres["mean_test_score"]
        results['std_accuracy'] = cvres["std_test_score"]
        results[hyperparams] = [list(params.values()) for params in cvres["params"]]
        results = results.sort_values('mean_accuracy', ascending=False)
        print(f'Top 5 models leading to the maximum accuracy on {n_cv} CV: ')
        print(results[:5].to_string())

    @staticmethod
    def print_metrics_crossval(score, n_cv):
        results = {'mean_accuracy': np.mean(score['test_accuracy']),
                   'std_accuracy': np.std(score['test_accuracy'])}
        results = pd.DataFrame.from_dict(results, orient='index')
        print(f'{n_cv} CV results: \n {results.T.to_string()}')

    @property
    def models(self):
        return {
            'rf': Pipeline([
                ('scaler', MinMaxScaler(feature_range=(0, 1))),
                ('clf', RandomForestClassifier(**self.best_params[self.model_type], random_state=self.seed, n_jobs=-1))
            ]),
        }

    @property
    def space_params(self):
        return {
            'rf': {
                'clf__max_depth': [2, 4, 6, 8],
                'clf__min_samples_split': [0.05, 0.1, 0.2],
                'clf__n_estimators': [100, 200, 300],
                'clf__max_features': [0.25, 0.5, 0.75],
                'clf__max_samples': [0.25, 0.5, 0.75],
            },
        }
