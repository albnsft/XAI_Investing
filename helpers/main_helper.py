from copy import deepcopy
from datetime import datetime

def add_env_args(parser, step_in_hour: float, ticker: str, dates: list):
    parser.add_argument("-sz", "--step_size", default=step_in_hour, help="Step size in hours.", type=float)
    parser.add_argument("-t", "--ticker", default=ticker, help="Specify stock ticker.", type=str)
    parser.add_argument("-n", "--normalisation_on", default=False, help="Normalise features.", type=bool)
    parser.add_argument("-starttrain", "--start_trading_train", default=dates[0], help="Start trading train.", type=datetime)
    parser.add_argument("-endtrain", "--end_trading_train", default=dates[1], help="End trading train.", type=datetime)
    parser.add_argument("-startva;", "--start_trading_val", default=dates[2], help="Start trading valid.", type=datetime)
    parser.add_argument("-endval", "--end_trading_val", default=dates[3], help="End trading valid.", type=datetime)
    parser.add_argument("-endtest", "--start_trading_test", default=dates[4], help="Start trading test.", type=datetime)
    parser.add_argument("-starttest", "--end_trading_test", default=dates[5], help="End trading test.", type=datetime)


def get_env_configs(args):
    train_env_config = {
        "ticker": args["ticker"],
        "start_trading": args["start_trading_train"],
        "end_trading": args["end_trading_train"],
        "step_size": args["step_size"],
        "normalisation_on": args["normalisation_on"],
    }

    valid_env_config = deepcopy(train_env_config)
    valid_env_config["start_trading"] = args["start_trading_val"]
    valid_env_config["end_trading"] = args["end_trading_val"]

    test_env_config = deepcopy(train_env_config)
    test_env_config["start_trading"] = args["start_trading_test"]
    test_env_config["end_trading"] = args["end_trading_test"]

    return train_env_config, valid_env_config, test_env_config


