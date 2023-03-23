from datetime import datetime, timedelta
import pandas as pd


def split_dates(split: float = None, start_date: datetime = None, end_date: datetime = None):
    start_train_date = start_date
    end_test_date = end_date
    end_train_date = (start_train_date + (end_test_date - start_train_date) * split).replace(hour=0, minute=0, second=0,
                                                                                             microsecond=0,
                                                                                             nanosecond=0)
    start_valid_date = end_train_date + timedelta(hours=24)
    end_valid_date = (start_valid_date + (end_test_date - start_train_date) * (1 - split) / 2).replace(hour=0, minute=0,
                                                                                                       second=0,
                                                                                                       microsecond=0,
                                                                                                       nanosecond=0)
    start_test_date = end_valid_date + timedelta(hours=24)
    return [start_train_date, end_train_date, start_valid_date, end_valid_date, start_test_date, end_test_date]


def informations(clf, importance: pd.DataFrame = None, columns: list = None):
    method = clf.method_type
    base = {f'accuracy_{method}': clf.report_val['accuracy'].values[0],
            f'outperformance_{method}': clf.report_val['outperformance'].values[0], }
    if method == 'baseline':
        new = {
            'model': clf.model_type,
            'ticker': clf.ticker,
        }
        base.update(new)
    if method != 'baseline' and importance is not None and len(importance) != 0 and not importance.empty:
        new = {
            "sum_importance": importance['feature_importance'].sum(),
            'removed_columns': list(importance.index),
        }
        base.update(new)
    return pd.DataFrame.from_dict(base, orient='index').T


def get_best_features(metrics_df: pd.DataFrame, metric: str = 'accuracy', method: str = None, all_columns: list = None):
    metrics = metrics_df.copy()
    label_baseline, label_pi = "{0}_baseline".format(metric), f"{metric}_{method}"
    metrics[f'{metric}_diff'] = (metrics[label_pi] - metrics[label_baseline]).astype(float)
    print('Increase of accuracy by removing features: ')
    print(metrics.sort_values([f'{metric}_diff'], ascending=[False])[[f'{metric}_diff', "removed_columns"]].iloc[:5].to_string())
    errors = metrics.sort_values(['ticker', f'{metric}_diff'], ascending=[True, False]) \
        .groupby(by=['ticker'], as_index=False).nth(0)
    print(f'For {errors["ticker"].values} removing {errors["removed_columns"].values[0]}'
          f' had the highest increased (by {errors[f"{metric}_diff"].values}) on the accuracy of validation set')
    removed_columns = errors["removed_columns"].values[0]
    columns = list(set(all_columns) - set(removed_columns))
    return columns, removed_columns