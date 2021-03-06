<h1 align="center">Clustered portfolio optimization</h1>
<p align="center">A scalable hierarchical risk parity optimizer</p>

## Theory

TODO

## Implemented features

- CLI
- Blended risk parity and sharpe optimization
- Periodic rebalancing
- Optimization on hierarchical cluster trees for stable diversification
- Gridsearch tuning of hyperparameters
- Simple backtest report 
- TODO refactor and document API so that it is cleaner and customizable

## CLI Usage

To install, run
```shell
git clone https://github.com/borab96/risk-parity.git
cd risk-parity
pip install .
```

Run ``rpp -h`` for a list of settings that can be specified. The only requirement is the specification of a list of symbols.
The help command returns 
```shell
usage: rpp [-h] [--cash CASH] [--period {1mo,3mo,6mo,1y,2y,5y,10y,ytd,max}]
           [--rebalance REBALANCE] [--save-fig SAVE_FIG] [--leverage LEVERAGE]
           [--short SHORT] [--tune TUNE TUNE TUNE TUNE] [--gamma GAMMA]
           [--cluster CLUSTER] [--browser-output BROWSER]
           list of symbols [list of symbols ...]

Portfolio optimizer.

positional arguments:
  list of symbols       symbols of holdings or equity universe

optional arguments:
  -h, --help            show this help message and exit
  --cash CASH           starting cash
  --period {1mo,3mo,6mo,1y,2y,5y,10y,ytd,max}
                        start date of lookback period
  --rebalance REBALANCE
                        how frequently should the portfolio be rebalanced
  --save-fig SAVE_FIG   If true saves figures in plots dir individually
  --leverage LEVERAGE   Leverage factor. No leverage by default
  --short SHORT         Allow short selling?
  --tune TUNE TUNE TUNE TUNE
                        Run grid search? arg[0]: size of gammas, arg[1:]
                        arguments of np.arange
  --gamma GAMMA         set gamma hyperparameter. gamma=1 is sharpe
                        maximization, gamma=0 is risk parity optimization
  --cluster CLUSTER     If true, does hierarchically clustered optimization
  --browser-output BROWSER
                        If true, displays figures on default browser

...
```

The hyperparameters of the optimization algorithm are ``gamma`` and ``rebalance``. The former
controls the convex combination of risk parity error minimization and Sharpe ratio
maximization and the latter specifies a holding period after which the optimizer is run again. The tuner seeks to 
maximize the Sharpe ratio. The reasoning is that the hyperparameter space represents a space of risk parity optimized 
solutions, at least for ``gamma<1/2`` which should now be ranked by the returns they offer. 

The backtester backfills positions after the optimal weights are computed. The so-called in-sample results assume
knowledge of the future time window. To get out-of-sample results, the positions have to shifted one window to the future
so that the trader positions themself based on the optimizer's result from the past window. Forecasting is not implemented.

``--gamma 1`` runs a Sharpe optimizer while 
``--gamma 0`` runs a risk parity optimizer

The ``--tune`` setting runs a gridsearch to optimize the Sharpe ratio among
optimal solutions. 

The ``--period`` setting sets the backtest period. 

### Example: Sector ETFs

As an example, imagine a portfolio constructed out of S&P 500 sector ETFs. The command
```shell
rpp XLC XLY XLP XLE XLF XLV XLI XLB XLRE XLK XLU --period 2y --tune 20 10 50 5 --cluster False
```
produces the output 

```shell
Optimal allocation of 10000.0
XLC: 1692.0266
XLY: 1480.0659
XLP: 594.0209
XLE: 126.8987
XLF: 122.0668
XLV: 727.311
XLI: 781.7656
XLB: 1100.4216
XLRE: 1763.8595
XLK: 867.966
XLU: 743.5974
---------------------------
CAGR 0.122
Average Sharpe ratio 1.815
Max drawdown ex. Covid 0.085


```

and saves the following 4 plots in the directory ``./plots`` (if ``--save-fig True``):

![](plots/sample_perf.png)
![](plots/sample_weights.png)
![](plots/sample_sharpes.png)
![](plots/sample_drawdown.png)

> The plotting backend has been updated to plotly. The applet now produces a combined [html output](https://htmlpreview.github.io/?https://github.com/borab96/risk-parity/blob/main/plots/combined_sample.html).

 In this case, the hyperparameter tuner chooses ``(gamma, rebalance )`` to be ``(0.37, 15)``
meaning that the optimal portfolio is one that is rebalanced every 15 trading days and one that gives slight preference 
to risk contribution diversification over maximizing the Sharpe ratio. The clustering algorithm is turned off because
our portfolio choice of SPY sectors is already hierarchical in nature. 

### Example: Many correlated assets

The file ``notebooks/top50.txt`` contains a list of the 50 largest US companies at the time it was saved (Jan 2022).
In this case we turn the clustering algorithm on to get more robust optimal portfolios. See [this notebook](https://nbviewer.org/github/borab96/risk-parity/blob/main/notebooks/clustering.ipynb)
for details. The hierarchical structure can be inferred from the plot below

![](plots/sample_corr.png)

The command
```shell
rpp notebooks/top50.txt --period 5y --cluster True --gamma 0 --rebalance 25 --browser True
```
runs the hierarchical optimizer on 25 day windows and produces a 5 year backtest report displayed in a new
tab on the default system browser.

> Because the clustering algorithm learns how to diversify based on correlation hierarchies, we don't really need
to enforce risk parity error minimization here. While not implemented, the optimization metric for each cluster could be chosen
to simply be returns themselves. 

 
