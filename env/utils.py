from datetime import timedelta

from database.HistoricalDatabase import HistoricalDatabase
from env.FinancialEnvironment import FinancialEnvironment


from pylab import plt
import numpy as np
import os


def env_creator(env_config, database: HistoricalDatabase = None):
    features = FinancialEnvironment.get_default_features(
            step_size=timedelta(hours=env_config["step_size"]),
            normalisation_on=env_config["normalisation_on"],
        )

    env = FinancialEnvironment(
        database=database,
        features=features,
        ticker=env_config["ticker"],
        step_size=timedelta(hours=env_config["step_size"]),
        start_of_trading=env_config["start_trading"],
        end_of_trading=env_config["end_trading"],
    )
    return env


def plot_final(
        ticker,
        algo_name,
        method_name,
        report_val,
        confusion_val,
        start_val,
        report_test,
        confusion_test,
        start_test,
        path,
        columns_removed: bool = None
):

    fig = plt.figure(constrained_layout=True, figsize=(15, 10))
    ax_dict = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        """
    )

    name = f"{ticker} - {algo_name} - {method_name}"
    if columns_removed is not None : name += f"\n columns removed: {columns_removed}"
    plt.suptitle(name)

    done_info = np.round(report_val[['accuracy', 'f1-score', 'auc', 'brier', 'outperformance']], 2)
    done_info_eval = np.round(report_test[['accuracy', 'f1-score', 'auc', 'brier', 'outperformance']], 2)

    table = ax_dict["A"].table(
        cellText=done_info.values,
        colLabels=done_info.columns,
        loc="center",
    )
    table.set_fontsize(10)
    #table.scale(0.5, 1.1)
    ax_dict["A"].set_axis_off()
    ax_dict["A"].title.set_text("Results validation")

    table = ax_dict["B"].table(
        cellText=done_info_eval.values,
        colLabels=done_info_eval.columns,
        loc="center",
    )
    table.set_fontsize(10)
    #table.scale(0.5, 1.1)
    ax_dict["B"].set_axis_off()
    ax_dict["B"].title.set_text("Results testing")

    confusion_val.plot(ax=ax_dict["C"])
    confusion_test.plot(ax=ax_dict["D"])

    start_val.plot(ax=ax_dict["E"], ylabel='$',
                      title=f'Agent vs Market Net Wealth')
    start_test.plot(ax=ax_dict["F"], ylabel='$',
                      title=f'Agent vs Market Net Wealth')

    pdf_filename = os.path.join(path, f"{algo_name}_{method_name}.jpg")
    # Write plot to pdf
    fig.savefig(pdf_filename)
    plt.close(fig)