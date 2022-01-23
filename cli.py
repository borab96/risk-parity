import argparse
from rpp.portfolio import Portfolio, grid_search
from rpp.plot_utils import combine_outs
import plotly.figure_factory as ff
import webbrowser
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Portfolio optimizer. ')
    parser.add_argument('symbols', metavar='list of symbols', type=str, nargs='+',
                        help='symbols of holdings or equity universe')
    parser.add_argument('--cash', dest='cash', action='store', type=float,
                        default=10000,
                        help='starting cash')
    parser.add_argument('--period', dest='period', action='store', type=str,
                        default='5y', choices=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                        help='start date of lookback period')
    parser.add_argument('--rebalance', dest='rebalance', action='store', type=float,
                        default=0,
                        help='how frequently should the portfolio be rebalanced')
    parser.add_argument('--save-fig', dest='save_fig', action='store', type=bool,
                        default=False,
                        help='If true saves figures in plots dir individually')
    parser.add_argument('--leverage', dest='leverage', action='store', type=float,
                        default=1.0,
                        help='Leverage factor. No leverage by default')
    parser.add_argument('--short', dest='short', action='store', type=bool,
                        default=False,
                        help='Allow short selling?')
    parser.add_argument('--tune', dest='tune', action='store', type=int,
                        default=False, nargs=4,
                        help='Run grid search? arg[0]: size of gammas, arg[1:] arguments of np.arange')
    parser.add_argument('--gamma', dest='gamma', action='store', type=float,
                        default=0.2,
                        help='set gamma hyperparameter. gamma=1 is sharpe maximization, '
                             'gamma=0 is risk parity optimization')
    parser.add_argument('--cluster', dest='cluster', action='store', type=bool,
                        default=True,
                        help='If true, does hierarchically clustered optimization')
    parser.add_argument('--browser-output', dest='browser', action='store', type=bool,
                        default=True,
                        help='If true, displays figures on default browser')
    args = parser.parse_args()
    try:
        with open(args.symbols[0], 'r') as f:
            args.symbols = f.read().splitlines()
            f.close()
    except FileNotFoundError:
        pass
    Pf = Portfolio(args=args)
    if args.tune:
        print("Running grid search")
        Pf.prog_bar_disable = True
        args.gamma, args.rebalance, Pf = grid_search(Pf, *args.tune)
        Pf.prog_bar_disable = False
        print(Pf.gamma, Pf.rebalance)
    else:
        Pf.optimize()
    if args.short:
        raise NotImplementedError("Short portfolios not implemented")

    figs = [Pf.plot_perf(),
            Pf.plot_corr(),
            Pf.plot_sharpes(),
            Pf.plot_weights(),
            Pf.plot_drawdown()]
            # ff.create_table(Pf.w_optim)]
    out = combine_outs(figs)
    if args.browser:
        webbrowser.open(out, new=2)
    Pf.summary()


if __name__ == '__main__':
    main()
