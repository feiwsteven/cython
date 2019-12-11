import numpy as np


# hazard function, two parameters weibull distribution

def hazard_weibull(x: np.float, psi: np.array) -> np.float:
    # 0< alpha < 1 in our model
    # beta > 0
    # x > 0
    beta = psi[0]
    alpha = psi[1]
    return beta * alpha * x ** (alpha - 1)


def surv_weibull(x: np.float, psi: np.array) -> np.float:
    beta = psi[0]
    alpha = psi[1]
    return np.exp(-beta * x ** alpha)


def cumulative_hazard_weibull(x, psi):
    beta = psi[0]
    alpha = psi[1]
    return beta * x ** alpha


def derivative_cumulative_hazard_weibull(x, psi):
    beta = psi[0]
    alpha = psi[1]
    res = np.zeros(2)
    # res[0] for beta
    res[0] = x ** alpha
    # res[1] for alpha
    res[1] = beta * alpha * x ** (alpha - 1)
    return res


def derivative_hazard_weibull(x, psi):
    beta = psi[0]
    alpha = psi[1]
    res = np.zeros(2)
    res[0] = alpha * x ** (alpha - 1)
    res[1] = beta * x ** (alpha - 1) + beta * (alpha - 1) * alpha * x ** (alpha - 2)
    return res
