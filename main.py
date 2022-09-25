import numpy as np
from scipy.stats import norm
import math


class Option:
    def __init__(self, r, S, K, T, sigma, type="C", t=0):
        self.r = r
        if S >= 0:
            self.S = S
        else:
            raise ValueError("Asset's price shouldn't be negative")
        if t >= 0:
            self.t = t
        else:
            raise ValueError("t should be a valid time")
        self.K = K
        if T >= self.t:
            self.T = T
        else:
            raise ValueError("T should be greater than t")
        if sigma >= 0:
            self.sigma = sigma
        else:
            raise ValueError("Sigma shouldn't be negative")
        if type == "C" or type == "P":
            self.type = type
        else:
            raise ValueError("Type should be put or call.")

    def get_tau(self):
        return self.T - self.t

    def get_d1(self):
        return (np.log(self.S/self.K) + (self.r + self.sigma**2/2)*self.get_tau())/(self.sigma*np.sqrt(self.get_tau()))

    def get_d2(self):
        return self.get_d1() - self.sigma*np.sqrt(self.get_tau())

    def get_payoff(self, S_t):
        if self.type == "C":
            return max(S_t - self.K, 0)
        else:
            return max(self.K - S_t, 0)
    
    def get_CRR_dt(self, M):
        return self.get_tau()/M
    
    def get_CRR_df(self, M):
        return math.exp(-self.r*self.get_CRR_dt(M))
    
    def get_CRR_u(self, M):
        return math.exp(self.sigma*math.sqrt(self.get_CRR_dt(M)))
    
    def get_CRR_d(self, M):
        return 1/self.get_CRR_u(M)
    
    def get_CRR_q(self, M):
        return (math.exp(self.r*self.get_CRR_dt(M) - self.get_CRR_d(M)))/(self.get_CRR_u(M) - self.get_CRR_d(M))
    
    def get_price(self, solver):
        if solver == "BSM":
            if self.type == "C":
                return self.S*norm.cdf(self.get_d1(), 0, 1) - self.K*np.exp(-self.r*self.get_tau())*norm.cdf(self.get_d2(), 0, 1)
            else:
                return self.K*math.exp(-self.r*self.get_tau())*norm.cdf(-self.get_d2(), 0, 1) - self.S*norm.cdf(-self.get_d1(), 0, 1)
            
    def get_delta(self):
        if self.type == "C":
            return norm.cdf(self.get_d1(), 0, 1)
        else:
            return -norm.cdf(-self.get_d1(), 0, 1)

    def get_gamma(self):
        return norm.pdf(self.get_d1(), 0, 1)/(self.S*self.sigma*np.sqrt(self.get_tau()))

    def get_theta(self):
        if self.type == "C":
            return -self.S*norm.pdf(self.get_d1(), 0, 1)*self.sigma/(2*np.sqrt(self.get_tau())) - self.r*self.K*math.exp(-self.r*self.get_tau())*norm.cdf(self.get_d2(), 0, 1)
        else:
            return self.K*math.exp(-self.r*self.get_tau())*norm.cdf(-self.get_d2(), 0, 1) - self.S*norm.cdf(-self.get_d1(), 0, 1)

    def get_vega(self):
        return self.S*norm.pdf(self.get_d1(), 0, 1)*np.sqrt(self.get_tau())

    def get_rho(self):
        if self.type == "C":
            return self.K*self.get_tau()*math.exp(-self.r*self.get_tau())*norm.cdf(self.get_d2(), 0, 1)
        else:
            return -self.K*self.get_tau()*math.exp(-self.r*self.get_tau())*norm.cdf(-self.get_d2(), 0, 1)
        
        