import argparse
from rpp.portfolio import Portfolio, grid_search


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
    parser.add_argument('--to_pdf', dest='to_pdf', action='store', type=bool,
                        default=False,
                        help='If true, prepares performance report')
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
                        default=0.2, nargs=1,
                        help='set gamma hyperparameter. gamma=1 is sharpe maximization, '
                             'gamma=0 is risk parity optimization')

    args = parser.parse_args()
    # print(args.metric)
    Pf = Portfolio(args)
    if args.tune:
        print("Running grid search")
        args.gamma, args.rebalance = grid_search(Pf, *args.tune)
        print(args.gamma, args.rebalance)
    if args.short:
        raise NotImplementedError("Short portfolios not implemented")
    Pf = Portfolio(args)
    Pf.optimize()
    Pf.plot_perf()
    Pf.plot_sharpes()
    Pf.plot_weights()
    Pf.plot_drawdown()
    Pf.summary()



if __name__ == '__main__':
    main()