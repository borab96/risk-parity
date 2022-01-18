import numpy as np
import scipy.optimize as sco


def slsqp(func, num_assets, args, bound, constraints, tol=1e-9):
    """

    SLSQP minimizer
     :param func: objective function to minimize
    :type func: function
    :param num_assets: number of assets in portfolio
    :type num_assets: int
    :param args: arguments beside portfolio weights
    :type args: tuple
    :param bound: tuple of uniform bounds: (-1., 1.) or (0., 1.)

    .. todo:: handle non-uniform bounds

    :param constraints: List of constraints,
    :type constraints: list of dicts
    :param tol: optimizer tolerance (1e-9)
    :type tol: float, optional
    :return: weights, optimal function vlaue, convergence status

    """
    # num_assets = len(args[0])
    bounds = tuple(bound for _ in range(num_assets))
    opt = sco.minimize(func, np.array(num_assets * [1. / num_assets, ]), args=args, method='SLSQP', bounds=bounds,
                       constraints=constraints, tol=tol)
    if opt.success:
        return opt.x, opt.fun, 1
    else:
        print("optimizer failed to converge")
        return opt.x, opt.fun, 0


optimizers = [slsqp]
