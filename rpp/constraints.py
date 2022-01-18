import numpy as np
# from scipy.optimize import NonlinearConstraint

def default():
    """
    Only implements constraint :math:`\sum_iw_i=1`+long only.

    :param *argv: ("eq" or "ineq", constraint function)
    :return: constraint dictionary
    """
    return ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                   {'type': 'ineq', 'fun': lambda w: w})



