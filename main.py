import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt


class Option:
    """
    A class that represent a financial option
    
    
    Attributes
    ----------
    r : float
       The risk-free rate
    S : float
       Price of the underlying asset
    K : float
       Strike price of the option
    T : float
       Time of option expiration, in years
    sigma : float
       Standard deviation of the asset's returns
    Type : str, default = "C"
       Type of the option. "C" stands for call, "P" stands for put.
    t : float, default = 0
       Time of the beginning of the contract, in years
    flag : str, default = "EUR"
       Flag of the option. "AME" stands for american, "EUR" stands for european.
       
     """
    def __init__(self, r, S, K, T, sigma, Type="C", t=0, flag = "EUR"):
        self.r = r
        if S >= 0:
            self.S = float(S)
        else:
            raise ValueError("Asset's price shouldn't be negative")
        if t >= 0:
            self.t = float(t)
        else:
            raise ValueError("t should be a valid time")
        self.K = float(K)
        if T >= self.t:
            self.T = float(T)
        else:
            raise ValueError("T should be greater than t")
        if sigma >= 0:
            self.sigma = float(sigma)
        else:
            raise ValueError("Sigma shouldn't be negative")
        if Type == "C" or Type == "P":
            self.Type = Type
        else:
            raise ValueError("Type should be put or call.")
        self.flag = flag

    def get_tau(self):
        """ 
        Gets the time maturity of the option, in years
        
        Parameters
        ----------
        
        Returns
        ----------
        float
           Time maturity of the option
        
        """
        return self.T - self.t

    def get_d1(self):
        """
        Gets the d1 of the option. For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        
        Returns
        ----------
        float
           The d1 of the option
           
        """
        return (np.log(self.S/self.K) + (self.r + self.sigma**2/2)*self.get_tau())/(self.sigma*np.sqrt(self.get_tau()))

    def get_d2(self):
        """
        Gets the d2 of the option. For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        
        Returns
        ----------
        float
           The d2 of the option
           
        """
        return self.get_d1() - self.sigma*np.sqrt(self.get_tau())

    def get_payoff(self, S_t):
        """
        Gets the payoff of the option.
        
        Parameters
        ----------
        S_t : float
           The price of the underlying asset at time t
           
        Returns
        ----------
        float
           The payoff of the option at time t
          
        """
        if self.Type == "C":
            return max(S_t - self.K, 0)
        else:
            return max(self.K - S_t, 0)
    
    def get_CRR_dt(self, N):
        """
        Following the notation of the Cox-Ross-Rubinstein model, gets the dt of the option. For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        N : int
           Number of binomial time steps
        
        Returns
        ----------
        float
           The dt of the option
        """
        return float(self.get_tau()/N)
    
    def get_CRR_df(self, N):
        """
        Following the notation of the Cox-Ross-Rubinstein model, gets the df of the option. For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        
        Parameters
        ----------
        N : int
           Number of binomial time steps
           
        Returns
        ----------
        float
           The df of the option
        """
        return math.exp(-self.r*self.get_CRR_dt(N))
    
    def get_CRR_u(self, N):
        """
        Following the notation of the Cox-Ross-Rubinstein model, gets the proportion that the price of the underlying asset goes up at each binomial time step (here, we call it "the proportion u of the option"). For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        
        Parameters
        ----------
        N : int
           Number of binomial time steps
           
        Returns
        ----------
        float
           The proportion u of the option

        """
        return math.exp(self.sigma*math.sqrt(self.get_CRR_dt(N)))
    
    def get_CRR_d(self, N):
        """
        Following the notation of the Cox-Ross-Rubinstein model, gets the proportion that the price of the underlying asset goes down at each binomial time step (here, we call it "the proportion d of the option"). For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        N : int
           Number of binomial time steps
           
        Returns
        ----------
        float
           The proportion d of the option
        """
        return 1/self.get_CRR_u(N)
    
    def get_CRR_p(self, N):
        """
        Following the notation of the Cox-Ross-Rubinstein model, gets the probability that the price of the underlying asset will go up at each binomial time step (here, we call it "the p of the option"). For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        N  : int
           Number of binomial time steps
           
        Returns
        ----------
        float
           The p of the option

        """
        return (math.exp(self.r*self.get_CRR_dt(N) - self.get_CRR_d(N)))/(self.get_CRR_u(N) - self.get_CRR_d(N))
    
    def get_price(self, solver, N = None):
        """
        Gets the price of the option, based on the solver. For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        solver : str
           Method used to estimate the price of the option. "BSM" stands for Black-Scholes-Merton and "CRR" stands for Cox-Ross-Rubinstein
        N : int, default = None
           Number of binomial time steps, used for binomial pricing models
           
        Returns 
        ----------
        float
           The price of the option
        """
        if solver == "BSM":
            if self.Type == "C":
                return self.S*norm.cdf(self.get_d1(), 0, 1) - self.K*np.exp(-self.r*self.get_tau())*norm.cdf(self.get_d2(), 0, 1)
            else:
                return self.K*math.exp(-self.r*self.get_tau())*norm.cdf(-self.get_d2(), 0, 1) - self.S*norm.cdf(-self.get_d1(), 0, 1)
        elif solver == "CRR":
            n_list = np.arange(0, (N + 1), 1)
            payoff_list = []
            for n in n_list:
                if self.Type == "C":
                    payoff = np.maximum(0, (self.S*self.get_CRR_u(N)**n*self.get_CRR_d(N)**(N - n) - self.K))   
                else:
                    payoff = np.maximum(0, -(self.S*self.get_CRR_u(N)**n*self.get_CRR_d(N)**(N - n) - self.K))
                payoff_list.append(payoff)
            for i in np.arange(N - 1, -1, -1):
                for j in np.arange(0, i + 1, 1):
                    if self.flag == "EUR":
                        payoff_list[j] = (self.get_CRR_p(N)*payoff_list[j + 1] + (1 - self.get_CRR_p(N))*payoff_list[j])*self.get_CRR_df(N)
                    else:
                        if self.type == "C":
                            payoff_list[j] = np.maximum((self.S*self.get_CRR_u(N)**j*self.get_CRR_d(N)**(i - j) - self.K),
                                                 (self.get_crr_p(N)*payoff_list[j + 1] + (1 - self.get_CRR_p(N))*payoff_list[j])*self.get_CRR_df(N))
                        else:
                            payoff_list[j] = np.maximum(-(self.S*self.get_CRR_u(N)**j*self.get_CRR_d(N)**(i - j) - self.K),
                                                 (self.get_crr_p(N)*payoff_list[j + 1] + (1 - self.get_CRR_p(N))*payoff_list[j])*self.get_CRR_df(N))
                            
            return payoff_list[0]
                
                    
            
            
   # def plot_convergence(step, max_time, min_time = 1, solver = "CRR"):
    #    if solver == "CRR":
    #        bsm_price = self.get_price(solver = "BSM")
    #        times = range(min_time, max_time, step)
    #        values = [self.get_price(solver, N) for N in times]
    #       plt.plot(times, values, label = "Values")
    #        plt.axhline(bsm_price, color = "r", ls = "dashed", label = "BSM price")
    #        plt.grid()
    #        plt.xlabel("Number of steps")
    #        plt.ylabel("European option value")
    #        plt.xlim(0, max_time)
    #        plt.legend(loc = 4)
    #        plt.show()
            
            
    def get_delta(self):
        """
        Gets the delta of the option.  For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        
        Returns
        ----------
        float
           The delta of the option
        """
        if self.flag == "EUR":
           if self.Type == "C":
               return norm.cdf(self.get_d1(), 0, 1)
           else:
               return -norm.cdf(-self.get_d1(), 0, 1)

    def get_gamma(self):
        """
        Gets the gamma of the option. For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        
        Returns
        ----------
        float
           The gamma of the option
        """
        if self.flag == "EUR":
           return norm.pdf(self.get_d1(), 0, 1)/(self.S*self.sigma*np.sqrt(self.get_tau()))

    def get_theta(self):
        """ 
        Gets the theta of the option. For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        
        Returns
        ----------
        float
           The theta of the option
        """
        if self.flag == "EUR":
           if self.type == "C":
              return -self.S*norm.pdf(self.get_d1(), 0, 1)*self.sigma/(2*np.sqrt(self.get_tau())) - self.r*self.K*math.exp(-self.r*self.get_tau())*norm.cdf(self.get_d2(), 0, 1)
           else:
              return self.K*math.exp(-self.r*self.get_tau())*norm.cdf(-self.get_d2(), 0, 1) - self.S*norm.cdf(-self.get_d1(), 0, 1)

    def get_vega(self):
        """ 
        Gets the vega of the option. For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        
        Returns
        ----------
        float
           The vega of the option.
        """
        if self.flag == "EUR":
           return self.S*norm.pdf(self.get_d1(), 0, 1)*np.sqrt(self.get_tau())

    def get_rho(self):
        """ 
        Gets the rho of the option. For more information, visit https://github.com/JulioManuelOsorio/derivatives-package
        
        Parameters
        ----------
        
        Returns
        ----------
        float
           The rho of the option.
        """
        if self.flag == "EUR":
           if self.Type == "C":
              return self.K*self.get_tau()*math.exp(-self.r*self.get_tau())*norm.cdf(self.get_d2(), 0, 1)
           else:
              return -self.K*self.get_tau()*math.exp(-self.r*self.get_tau())*norm.cdf(-self.get_d2(), 0, 1)
        
        
        