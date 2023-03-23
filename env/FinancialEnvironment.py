from features.Features import *
from database import HistoricalDatabase


class FinancialEnvironment:
    def __init__(
            self,
            database: HistoricalDatabase = None,
            features: list = None,
            ticker: str = None,
            step_size: timedelta = None,
            start_of_trading: datetime = None,
            end_of_trading: datetime = None,
            normalisation_on: bool = None,
            verbose: bool = False
    ):

        self.database = database
        self.ticker = ticker
        self.step_size = step_size
        self.start_of_trading = start_of_trading
        self.end_of_trading = end_of_trading
        self.verbose = verbose
        self.features = features or self.get_default_features(step_size, normalisation_on)
        self.max_feature_window_size = max([feature.window_size for feature in self.features])
        self.state: State = None
        self._check_params()
        self.init()

    def init(self) -> np.ndarray:
        now_is = self.start_of_trading  # (self.max_feature_window_size + self.step_size * self.n_lags_feature)
        self.state = State(market=self._get_market_data(now_is), now_is=now_is)
        data, dates = [], [self.state.now_is]
        while self.end_of_trading >= self.state.now_is:
            self._forward()
            data.append(self.get_features())
            dates.append(self.state.now_is)
        self.data = pd.DataFrame(data, index=dates[:-1], columns=[feature.name for feature in self.features]).dropna()
        self.data = self.data[self.data['Direction_1']!=0] #for binary classification, 0 being outsider
        self.y = pd.DataFrame(np.where(self.data['Direction_1'].iloc[1:]==1, 1, 0), columns=['label'], index=self.data.iloc[1:].index)
        self.X = self.data.shift(1).iloc[1:]

    def _forward(self):
        self._update_features()
        self.update_internal_state()

    def _update_features(self):
        for feature in self.features:
            feature.update(self.state)

    def get_features(self) -> np.ndarray:
        return np.array([feature.current_value for feature in self.features])

    def update_internal_state(self):
        self.state.now_is += self.step_size
        if self.state.now_is not in self.database.calendar[self.ticker] and self.state.now_is <= self.end_of_trading:
            self.state.now_is = self.database.get_next_timestep(self.state.now_is, self.ticker)
        self.state.market = self._get_market_data(self.state.now_is)

    def _get_market_data(self, datepoint: timedelta):
        data = self.database.get_last_snapshot(datepoint, self.ticker)
        return Market(**{k.lower(): v for k, v in data.to_dict().items()})

    def _check_params(self):
        assert self.start_of_trading <= self.end_of_trading, "Start of trading Nonsense"

    @staticmethod
    def get_default_features(step_size: timedelta, normalisation_on: bool):
        return [Return(update_frequency=step_size,
                        lookback_periods=1,
                        name="Return_1",
                        normalisation_on=normalisation_on),
                Direction(update_frequency=step_size,
                          lookback_periods=1,
                          name="Direction_1",
                          normalisation_on=normalisation_on),
                Direction(update_frequency=step_size,
                          lookback_periods=2,
                          name="Direction_2",
                          normalisation_on=normalisation_on),
                Direction(update_frequency=step_size,
                          lookback_periods=3,
                          name="Direction_3",
                          normalisation_on=normalisation_on),
                Direction(update_frequency=step_size,
                          lookback_periods=5,
                          name="Direction_5",
                          normalisation_on=normalisation_on),
                Direction(update_frequency=step_size,
                          lookback_periods=21,
                          name="Direction_21",
                          normalisation_on=normalisation_on),
                Volatility(update_frequency=step_size,
                           normalisation_on=normalisation_on),
                RSI(update_frequency=step_size,
                    normalisation_on=normalisation_on),
                EWMA(lookback_periods=7,
                     name="EWMA_7",
                     update_frequency=step_size,
                     normalisation_on=normalisation_on),
                EWMA(lookback_periods=14,
                     update_frequency=step_size,
                     name="EWMA_14",
                     normalisation_on=normalisation_on),
                EWMA(lookback_periods=21,
                     update_frequency=step_size,
                     name="EWMA_21",
                     normalisation_on=normalisation_on),
                MACD(update_frequency=step_size,
                     normalisation_on=normalisation_on),
                ATR(update_frequency=step_size,
                    normalisation_on=normalisation_on),
                STOCH(update_frequency=step_size,
                      normalisation_on=normalisation_on),
                WilliamsR(update_frequency=step_size,
                          normalisation_on=normalisation_on),
                OBV(update_frequency=step_size,
                    normalisation_on=normalisation_on),
                ChaikinFlow(update_frequency=step_size,
                            normalisation_on=normalisation_on)]

    def backtest(self, predictions: np.ndarray = None, spread: float = 0.00006, cash: float = 100,
                 type_env: str = 'valid', verbose: bool = True, type_algo: str = 'rf'):
        data = self.data.copy()
        data['position'] = 0 #first timestep position is neutral as data t-1 required
        data['position'].iloc[1:] = np.where(predictions > 0.5, 1, -1)

        def ptf_base(start_sum, rets):
            for r in rets:
                v = start_sum * (1 + r)
                yield v
                start_sum = v

        data['strategy'] = data['position'] * data['Return_1'] # Calculates the strategy returns given the position values
        # determine when a trade takes place
        trades = data['position'].diff().fillna(1) != 0
        # instantiate strategy with transaction cost
        data['strategy_tc'] = data['strategy']
        # spread = 0.00006 --> bid-ask spread on professional level
        tc = spread / self.database.data[self.ticker]['Close'].mean()
        # subtract transaction costs from return when trade takes place
        data['strategy_tc'][trades] -= tc
        # compute the VL base 100 of the passive returns, strategy and strategy with transaction cost
        data['cum_returns'] = pd.Series(list(ptf_base(cash, data['Return_1'])), index=data.index)
        data['cum_strategy'] = pd.Series(list(ptf_base(cash, data['strategy'])), index=data.index)
        data['cum_strategy_tc'] = pd.Series(list(ptf_base(cash, data['strategy_tc'])), index=data.index)
        VL_strat = data['cum_strategy_tc'].iloc[-1]
        aperf = VL_strat / cash - 1
        operf = aperf - (data['cum_returns'].iloc[-1] / cash - 1)
        if verbose:
            print(f'************************ On {type_env} set *********************************')
            print(f'The number of trades is {sum(trades)}, there is a total of {len(data)} ticks')
            print('The absolute performance of the strategy with tc is {:.1%}'.format(aperf))
            print('The outperformance of the strategy with tc is {:.1%}'.format(operf))
            print(100 * '*')
        plot = data[['cum_returns', 'cum_strategy_tc']]
        return plot, operf