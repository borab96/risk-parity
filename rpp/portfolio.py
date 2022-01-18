import pandas as pd
import numpy as np
import rpp.optimizers as opt
import rpp.metrics as metrics
import yfinance as yf
from matplotlib import rcParams
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
import matplotlib.pyplot as plt
from tqdm import tqdm


def grid_search(Pf, *args):
    gammas = np.linspace(0, 1, args[0])
    rebalances = np.arange(args[1], args[2], args[3])
    metric = np.zeros([len(gammas), len(rebalances)])
    for ig in tqdm(range(len(gammas))):
        for ir, rebalance in enumerate(rebalances):
            Pf.gamma = gammas[ig]
            Pf.rebalance = rebalance
            Pf.optimize()
            if Pf.sharpe > metric.max():
                gamma_opt, rebalance_opt = gammas[ig], rebalance
            metric[ig, ir] = Pf.sharpe
    return gamma_opt, rebalance_opt


class Portfolio:
    def __init__(self,
                 *symbols,
                 period='2y',
                 rebalance=15,
                 leverage=1.0,
                 cash=10000,
                 gamma=0.2,
                 risk_free_rate=0.0,
                 args=False):
        self.args = args if args else None
        self.symbols = args.symbols if args else symbols
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]
        self.period = args.period if args else period
        self.rebalance = int(args.rebalance) if args else rebalance
        self.leverage = args.leverage if args else leverage
        self.cash = args.cash*self.leverage if args else cash*self.leverage
        self.gamma = args.gamma if args else gamma
        self.risk_free_rate = risk_free_rate
        self.data = pd.DataFrame(columns=self.symbols)
        self.trade_days = 252
        self.optimizer = opt.slsqp
        self.metric = metrics.rcp_error
        self.constraint = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                           {'type': 'ineq', 'fun': lambda w: w})
        self.w_optim = None
        self.rebalance_index = []
        self.w_optims = []
        self.sharpes = []
        self.risk_budget = np.array(len(self.symbols)*[1./len(self.symbols), ])
        self.collect_data()
        self.position = None

    def collect_data(self):
        for t in self.symbols+["SPY"]:
            ticker = yf.Ticker(t)
            close = ticker.history(period=self.period).Close#.rolling(1).mean()
            if t == "SPY":
                self.benchmark = np.log(close).diff().dropna()
                self.benchmark.name = "benchmark"
            else:
                self.data[t] = np.log(close).diff().dropna()

    @property
    def mean_market_return(self):
        return self.benchmark.mean()*self.trade_days

    @property
    def mean_return(self):
        returns = self.data.mean()*self.trade_days
        return returns

    def capm_return(self, data_):
        returns = data_.copy()
        returns = returns.join(self.benchmark, how="left")
        cov = returns.cov()
        betas = (cov["benchmark"] / cov.loc["benchmark", "benchmark"]).drop("benchmark")
        benchmark_mean = returns["benchmark"].mean()*self.trade_days
        returns = self.risk_free_rate + betas * (benchmark_mean - self.risk_free_rate)
        return returns

    def optimize(self):
        if not self.rebalance:
            self._minimizer(self.data)
            self.position = pd.DataFrame(np.tile(self.w_optim, (len(self.data.index), 1)),
                                         columns=self.symbols, index=self.data.index)
        elif self.rebalance > 0:
            self.position = pd.DataFrame(0, columns=self.symbols, index=self.data.index)
            for i in range(self.rebalance, len(self.data), self.rebalance):
                window = self.data[self.symbols].iloc[i - self.rebalance:i]  # use data from self.rebalance days prior
                self._minimizer(window)
                self.position.loc[window.index] = self.w_optim
        else:
            raise ValueError(f"{self.rebalance} cannot be interpreted as a holding duration")

    def _minimizer(self, data):
        cov = data.cov() * self.trade_days
        self.w_optim = self.optimizer(self.metric, len(self.symbols),
                                      (self.capm_return(data), cov, self.risk_budget, self.gamma), (0, 1),
                                      self.constraint)[0]
        self.w_optims.append(self.w_optim)
        self.sharpes.append(-metrics.sharpe(self.w_optim, self.capm_return(data), cov, 0.01))
        self.w_optims.append(self.w_optim)
        self.rebalance_index.append(data.index[-1])

    @property
    def portfolio_returns(self):
        return (self.position.shift(0)*self.data).sum(axis=1)

    @property
    def portfolio_value(self):
        return self.portfolio_returns.cumsum().apply(np.exp)

    @property
    def benchmark_value(self):
        return self.benchmark.cumsum().apply(np.exp)

    @property
    def benchmark_drawdown(self):
        return self.benchmark_value/self.benchmark_value.rolling(self.trade_days).max()-1.

    @property
    def portfolio_drawdown(self):
        return self.portfolio_value/self.portfolio_value.rolling(self.trade_days).max()-1.

    @property
    def cagr(self):
        years_elapsed = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days / self.trade_days
        return (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) ** (1 / years_elapsed) - 1

    @property
    def sharpe(self):
        return self.portfolio_returns.mean()*self.trade_days/(self.portfolio_returns.std()*np.sqrt(self.trade_days))

    def plot_sharpes(self):
        plt.figure()
        plt.plot(self.rebalance_index, self.sharpes)
        plt.plot(self.rebalance_index, np.mean(self.sharpes)*np.ones(len(self.sharpes)))
        plt.xticks(rotation=25)
        plt.grid()
        plt.savefig('plots/sharpes.png')
        plt.close()

    def plot_perf(self):
        plt.figure()
        if self.rebalance:
            plt.plot(self.portfolio_value, label='portfolio')
            plt.plot(self.benchmark_value, label='benchmark')
        else:
            plt.plot((self.w_optim*self.data).sum(axis=1).cumsum().apply(np.exp), label='portfolio')
            plt.plot(self.benchmark.cumsum().apply(np.exp), label='benchmark')
        plt.xticks(rotation=25)
        plt.legend()
        plt.title(f"In sample portfolio performance - rebalanced every {self.rebalance} days")
        plt.savefig('plots/perf.png')
        plt.close()

    def plot_weights(self):
        plt.figure()
        y = []
        for i, sym in enumerate(self.symbols):
            y.append(np.array(self.w_optims)[:, i])
            if i:
                plt.bar(np.arange(len(self.w_optims)), y[i], label=sym, bottom=sum(y)-y[-1])
            else:
                plt.bar(np.arange(len(self.w_optims)), y[0], label=sym)
            plt.legend()
            plt.grid()
        plt.title(f"In portfolio weight - rebalanced every {self.rebalance} days")
        plt.savefig('plots/weights.png')
        plt.close()

    def plot_drawdown(self):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_drawdown.rolling(10).min(), label="portfolio")
        plt.plot(self.benchmark_drawdown.rolling(10).min(), label='benchmark')
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.title(f"Portfolio drawdown - rebalanced every {self.rebalance} days")
        plt.grid()
        plt.legend()
        plt.subplot(2, 1, 2)
        diff = self.portfolio_drawdown.rolling(30).min()-self.benchmark_drawdown.rolling(30).min()
        plt.plot(diff, label="portfolio-benchmark")
        plt.axhline(diff.mean(), color='r')
        plt.legend()
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.grid()
        plt.savefig('plots/drawdown.png')
        plt.close()

    def summary(self):
        print(f"Optimal allocation of {self.cash}")
        for i, symbol in enumerate(self.symbols):
            print(f"{symbol}: {round(self.w_optim[i]*self.cash, 4)}")
        print("---------------------------")
        print(f"CAGR {round(self.cagr, 3)}")
        print(f"Average Sharpe ratio {round(np.mean(self.sharpes), 3)}")


