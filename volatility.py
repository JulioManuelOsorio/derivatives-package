import numpy as np
from scipy.stats import norm
import math
import pandas as pd
from scipy.optimize import brentq, fsolve


def log_returns(price):
    """ 
    Gets the log returns series given a price time series
        
    Parameters
    ----------
    price : list or pandas.Series
        The price time series
    
    Returns
    ----------
    pandas.Series
        The log returns series
    """
    if type(price) == list:
        return np.log(1 + pd.Series(price).pct_change())
    else:
        return np.log(1 + price.pct_change())

def mean_log_returns(price):
    """
    Gets the mean of the log returns series of a given price time series
    
    Parameters
    ----------
    price : list or pandas.Series
        The price time series
    
    Returns
    ----------
    float
        The mean of the log returns series of the price time series
    
    """
    log_return = log_returns(price)
    return np.mean(log_return.dropna())

def historical_vol(price):
    """
    Gets the historical volatility (std) of a given price time series
    
    Parameters
    ----------
    price : list or pandas.Series
        The price time series
    
    Returns
    ----------
    float
        The historical volatility of the price time series
    
    """
    log_return = log_returns(price).dropna()
    mean_log_return = mean_log_returns(price)
    return np.sqrt(sum([(r - mean_log_return)**2 for r in log_return])/(len(log_return) - 1))

def skew(price):
    """
    Gets the skewness of a given price time series

    Parameters
    ----------
    price : list or pandas.Series
        The price time series

    Returns
    -------
    float
        The skewness of the price time series
    
    """
    log_return = log_returns(price).dropna()
    mean_log_return = mean_log_returns(price)
    return (sum([(r - mean_log_return)**3 for r in log_return])/(1/len(log_return)*(sum([(r - mean_log_return)**2 for r in log_return]))**3/2))

def kurt(price):
    """
    Gets the kurtosis of a given price time series

    Parameters
    ----------
    price : list or pandas.Series
        The price time series

    Returns
    -------
    float
        The kurtosis of the price time series
    
    """
    log_return = log_returns(price).dropna()
    mean_log_return = mean_log_returns(price)
    return ((1/len(log_return))*sum([(r - mean_log_return)**4 for r in log_return]))/(((1/len(log_return))*sum([(r - mean_log_return)**2 for r in log_return**2]))) - 3

def implied_vol(S, K, r, T, C, op_type,  t = 0, a = 0, b = 2, xtol = 1e-6, solver = "brentq"):
   _S, _K, _r, _t, _T, _C, _op_type = S, K, r, t, T, C, op_type
   def BSM(_S, _K, _r, _t, sigma, _op_type):
       tau = _T - _t
       d1 = np.log(_S/_K) + (_r + sigma**2/2)*tau/(sigma*np.sqrt(tau))
       d2 = d1 - sigma*np.sqrt(tau)
       if _op_type == "C":
           return norm.cdf(d1, 0, 1) - math.exp(- _r*tau)*_K*norm.cdf(d2, 0, 1)
       else:
           return _K*math.exp(-_r*tau)*norm.cdf(d2, 0, 1) - _S*norm.cdf(-d1, 0, 1)
       
       if solver == "brentq":
        def implied_vol_obj_f(_S, _K, _r, _t, sigma, _C, _op_type):
                return _C - BSM(_S, _K, _r, _t, sigma, _op_type)
        def fcn(sigma):
            return implied_vol_obj_f(_S, _K, _r, _t, sigma, _C, _op_type)
            return brentq(fcn, a = a, b = b, xtol = xtol)
    
       
            