import pandas as pd
import numpy as np
import rpp.optimizers as opt
import rpp.utils as util
import rpp.metrics as metrics
import rpp.plot_utils as plt_
import yfinance as yf
from tqdm import tqdm as tqdm_cli
from tqdm.notebook import tqdm as tqdm_nb
import copy
# from functools import cached_property # Need python 3.9


def grid_search(Pf, *args):
    gammas = np.linspace(0, 1, args[0])
    rebalances = np.arange(args[1], args[2], args[3])
    metric = np.zeros([len(gammas), len(rebalances)])
    for ig in tqdm_cli(range(len(gammas))):
        for ir, rebalance in enumerate(rebalances):
            Pf.reset()
            Pf.gamma = gammas[ig]
            Pf.rebalance = rebalance
            Pf.optimize()
            if Pf.sharpe > metric.max():
                gamma_opt, rebalance_opt = gammas[ig], rebalance
                Pf_opt = copy.deepcopy(Pf)
            metric[ig, ir] = Pf.sharpe
    return gamma_opt, rebalance_opt, Pf_opt


class Portfolio:
    def __init__(self,
                 *symbols,
                 period='2y',
                 rebalance=15,
                 leverage=1.0,
                 cash=10000,
                 gamma=0.2,
                 risk_free_rate=0.0,
                 nb=True,
                 save_fig=False,
                 cluster=True,
                 args=False):
        self.args = args if args else None
        self.symbols = args.symbols if args else list(symbols)
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]
        self.period = args.period if args else period
        self.rebalance = int(args.rebalance) if args else rebalance
        self.leverage = args.leverage if args else leverage
        self.cash = args.cash*self.leverage if args else cash*self.leverage
        self.gamma = args.gamma if args else gamma
        self.nb = False if args else nb
        self.tqdm = tqdm_nb if self.nb else tqdm_cli
        self.save_fig = args.save_fig if args else save_fig
        self.cluster = args.cluster if args else cluster
        self.risk_free_rate = risk_free_rate
        self.data = pd.DataFrame(columns=self.symbols)
        self.trade_days = 252
        self.optimizer = opt.slsqp
        self.metric = metrics.rcp_error
        self.constraint = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                           {'type': 'ineq', 'fun': lambda w: w})
        self.collect_data()
        self.reset()
        self.covid = pd.date_range("2020-02-01", "2020-06-01", freq="B")
        self.prog_bar_disable = False

    def reset(self):
        self.w_optim = None
        self.rebalance_index = []
        self.w_optims = []
        self.sharpes = []
        self.risk_budget = np.array(len(self.symbols) * [1. / len(self.symbols), ])
        self.position = None

    def collect_data(self):
        print("Donwloading ticker data")
        for t in self.tqdm(self.symbols+["SPY"]):
            ticker = yf.Ticker(t.replace(" ", ""))
            close = ticker.history(period=self.period).Close#.rolling(1).mean()
            if t == "SPY":
                self.benchmark = np.log(close).diff().dropna()
                self.benchmark.name = "benchmark"
            else:
                if not close.isnull().sum():
                    self.data[t] = np.log(close).diff().dropna()
                else:
                    print(f"Dropping {t}, not enough data")

    @property
    def covariance(self):
        if len(self.data) > 1:
            return self.data.cov()*self.trade_days
        else:
            raise ValueError("data should not be None - something went wrong with data fetching")

    @property
    def correlation(self):
        if len(self.data) > 1:
            return self.data.corr()
        else:
            raise ValueError("data should not be None - something went wrong with data fetching")

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
            if self.cluster:
                self.cluster_minimizer(self.data)
            else:
                self.minimizer(self.data)
            self.position = pd.DataFrame(np.tile(self.w_optim, (len(self.data.index), 1)),
                                         columns=self.symbols, index=self.data.index)
        elif self.rebalance > 0:
            self.position = pd.DataFrame(0, columns=self.symbols, index=self.data.index)
            print(f"Running optimizer on windowed data with gamma={self.gamma}, rebalance={self.rebalance}, "
                  f"clustering={self.cluster} \n")
            for i in self.tqdm(range(self.rebalance, len(self.data), self.rebalance), disable=self.prog_bar_disable):
                window = self.data[self.symbols].iloc[i - self.rebalance:i]  # use data from self.rebalance days prior
                if self.cluster:
                    self.cluster_minimizer(window)
                else:
                    self.minimizer(window)
                self.position.loc[window.index] = self.w_optim.values if isinstance(self.w_optim, pd.Series) \
                    else self.w_optim
        else:
            raise ValueError(f"{self.rebalance} cannot be interpreted as a holding duration")

    def cluster_minimizer(self, data):
        cov = data.cov() * self.trade_days
        corr = data.corr()
        cluster_tree = util.linkage(corr, tree=True)
        sorted_idx = corr.iloc[cluster_tree.pre_order()].index
        w = pd.Series(1, index=sorted_idx)
        clusters = [sorted_idx]
        while len(clusters) > 0:
            clusters = [i[j:k] for i in clusters for j, k in ((0, int(len(i) / 2)), (int(len(i) / 2), len(i))) if
                        len(i) > 1]
            for i in range(0, len(clusters), 2):
                leaf0, leaf1 = clusters[i], clusters[i + 1]
                cov0, cov1 = cov.loc[leaf0, leaf0], cov.loc[leaf1, leaf1]
                clv = []
                for cov_, leaf in zip([cov0, cov1], [leaf0, leaf1]):
                    risk_budget_ = np.array(len(cov_) * [1. / len(cov_), ])
                    w_ = self.optimizer(self.metric, len(cov_),
                                      (self.capm_return(data[leaf]), cov_, risk_budget_, self.gamma), (0, 1),
                                      self.constraint)[0]
                    clv.append(w_ @ cov_ @ w_.T)
                a = 1 - clv[0] / (clv[1] + clv[0])
                w[leaf0] *= a
                w[leaf1] *= 1 - a
        self.w_optim = w
        self.w_optims.append(self.w_optim)
        self.sharpes.append(-metrics.sharpe(self.w_optim, self.capm_return(data), cov, 0.01))
        self.rebalance_index.append(data.index[-1])

    def minimizer(self, data):
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
        _ = self.portfolio_returns.cumsum().apply(np.exp)
        _.name = "portfolio"
        return _

    @property
    def benchmark_value(self):
        return self.benchmark.cumsum().apply(np.exp)

    @property
    def benchmark_drawdown(self):
        return self.benchmark_value/self.benchmark_value.rolling(30).max()-1.

    @property
    def portfolio_drawdown(self):
        return self.portfolio_value/self.portfolio_value.rolling(30).max()-1.

    @property
    def cagr(self):
        years_elapsed = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days / self.trade_days
        return (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) ** (1 / years_elapsed) - 1

    @property
    def sharpe(self):
        return self.portfolio_returns.mean()*self.trade_days/(self.portfolio_returns.std()*np.sqrt(self.trade_days))

    def plot_sharpes(self):
        fig = plt_.plot(x=self.rebalance_index, y=self.sharpes)
        fig.add_hline(y=np.mean(self.sharpes))
        if self.save_fig:
            fig.write_html('plots/sharpes.html')
        return fig

    def plot_perf(self):
        if self.rebalance:
            fig = plt_.plot(pd.concat([self.benchmark_value, self.portfolio_value], axis=1),
                            title=f"In sample portfolio performance - rebalanced every {self.rebalance} days")
        else:
            pf_ = (self.w_optim*self.data).sum(axis=1).cumsum().apply(np.exp)
            pf_.name = "portfolio"
            fig = plt_.plot(pd.concat([self.benchmark.cumsum().apply(np.exp), self.portfolio_value], axis=1),
                            title=f"In sample portfolio performance - no rebalancing")
        if self.save_fig:
            fig.write_html('plots/perfs.html')
        return fig

    def plot_weights(self):
        df = pd.DataFrame(self.w_optims, columns=self.symbols)
        fig = plt_.bar(df, title="Weights", barmode='stack')
        if self.save_fig:
            fig.write_html('plots/weights.html')
        return fig

    def plot_corr(self):
        fig = plt_.full_heat_map(self.correlation)
        if self.save_fig:
            fig.write_html('plots/clustered_correlations.html')
        return fig

    def plot_drawdown(self):
        dds = pd.concat([self.benchmark_drawdown.rolling(10).min(), self.portfolio_drawdown.rolling(10).min()], axis=1)
        fig = plt_.plot(dds, title="Max drawdown")
        if self.save_fig:
            fig.write_html('plots/drawdown.html')
        return fig

    def summary(self):
        print(f"Optimal allocation of {self.cash}")
        for i, symbol in enumerate(self.symbols):
            print(f"{symbol}: {round(self.w_optim[i]*self.cash, 4)}")
        print("---------------------------")
        print(f"CAGR {round(self.cagr, 3)}")
        print(f"Average Sharpe ratio {round(np.mean(self.sharpes), 3)}")
        dd = self.portfolio_drawdown
        print(f"Max drawdown ex. Covid {round(-1*np.min(dd.loc[~dd.index.isin(self.covid)]), 3)}")


