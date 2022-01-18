import numpy as np


def rcp_error(w, mean, cov, risk_budget, gamma=0.2):
    """
    risk contribution error compared to target.

    :param w:
    :param cov: pd.DataFrame annualized covariance
    :param risk_budget: desired risk distribution in the portfolio
    :return:
    """
    w = np.array(w)
    cov = cov.values
    portfolio_risk = np.sqrt(w.T.dot(cov.dot(w)))
    rc = w*(cov.dot(w))/portfolio_risk
    err = np.sum((rc-portfolio_risk*risk_budget)**2)
    return (1-gamma)*err +gamma*sharpe(w, mean, cov)


def sharpe(w, mean, cov, benchmark_rate=0.02):
    """
    Computes: :math:`-(r_{p} - r_{bm})/\sigma_p`

    :param w: asset weights
    :param cov: annaulized portfolio covariance
    :param mean: annualized portfolio mean returns
    :param benchmark_rate: risk free rate or some other benchmark
    :return: sharpe ratio
    :rtype: float
    """
    annualized_std = np.sqrt(w.T.dot(cov).dot(w))
    annualized_mean_return = mean.dot(w)
    return -(annualized_mean_return - benchmark_rate)/annualized_std
