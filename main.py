import argparse
from env.utils import env_creator
from helpers.main_helper import add_env_args, get_env_configs
from datetime import timedelta
from utils.utils import *
from database.HistoricalDatabase import HistoricalDatabase
from models.Model import Classifier
from selectfeatures import *
from postxplaination.postxplaination import *

import pandas as pd

pd.options.mode.chained_assignment = None  # Ignore Setting With Copy Warning


if __name__ == '__main__':

    #stocks to conduct the analysis on
    tickers = ['AAPL', 'GOOG']
    name = 'tech_stocks'
    #granularity of time, predicition are made every day
    step_size_in_hour = "24"
    #instantiate database
    database = HistoricalDatabase(tickers, name)

    #instantiate feature selection methods
    feature_selection_methods = {
                    'permutation_importance': PISelector(),
                    'impurity_importance': RFFeatureImportanceSelector(),
                    'lime_importance': LIMEFeatureImportanceSelector(),
                    'shape_importance': SHAPFeatureImportanceSelector(),
    }
    #instantiate black box model
    models = ['rf']

    #loop on each stock in the databse, there Apple and Google
    for ticker in tickers:

        #define the configuration
        dates = split_dates(split=0.7, start_date=database.start_date[ticker], end_date=database.end_date[ticker])
        parser = argparse.ArgumentParser(description="")
        add_env_args(parser, step_size_in_hour, ticker, dates)
        args = vars(parser.parse_args())
        train_env_config, valid_env_config, test_env_config = get_env_configs(args)

        #instantiate training, validation and testing environment
        train_env = env_creator(train_env_config, database)
        valid_env = env_creator(valid_env_config, database)
        test_env = env_creator(test_env_config, database)

        #full set of features
        all_columns = list(train_env.X.columns)
        #
        outperformance_baseline, best_clf = -1, None

        #loop on each model, there only random forest
        for model in models:

            #train the baseline classifier, find best hyperparams via time series split CV on train_env, save validation and test accuracy without using these environments
            clf_baseline = Classifier(train_env, valid_env, test_env, model_type=model, method_type='baseline', ticker=ticker,
                             verbose=True)
            clf_baseline.learn()
            base_info = informations(clf_baseline)

            # loop on each method to compute feature importance/selection
            for name_trf, method in feature_selection_methods.items():
                all_metrics = pd.DataFrame()
                for importance, columns in method.fit_transform(clf_baseline, train_env.X, train_env.y, valid_env.X, valid_env.y):
                    #find space of removable features by computing feature importance on valid_env
                    #for each combination of removable features, fit model without these features save validation acccuracy to compare with baseline
                    clf_find_features = Classifier(train_env, valid_env, test_env=None, model_type=model, method_type=name_trf,
                                      ticker=ticker, columns=columns, removed_columns=list(importance.index))
                    clf_find_features.learn()
                    new_info = informations(clf_find_features, importance, columns)
                    all_metrics = all_metrics.append(pd.concat([base_info, new_info], axis=1))

                all_metrics.reset_index(drop=True, inplace=True)
                #find best set of removed features i.e. the one leading to the highest accuracy increase on valid_env
                best_columns, removed_columns = get_best_features(all_metrics, metric='accuracy', method=name_trf, all_columns=all_columns)
                #train best model and evaluate it on test_env
                clf_test_features = Classifier(train_env, valid_env, test_env, model_type=model, method_type=name_trf,
                                 ticker=ticker, verbose=True, columns=best_columns, removed_columns=removed_columns)
                outperformance = clf_test_features.learn(test_info=True)

                if outperformance > outperformance_baseline:
                    #save the method leading to the highest outperformance on test set
                    outperformance_baseline = outperformance
                    best_clf = clf_test_features

            shap_plots(best_clf)
            shap_plots(clf_baseline)
        print('end')





