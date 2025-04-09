
"""
This class represents a single user arriving to the charging facility 
Author(s):  
            Cesar Santoyo
"""
from prettytable import PrettyTable
from scipy.integrate import quad, dblquad
from matplotlib import cm
from scipy.optimize import fmin
from scipy import optimize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import sys

class User:
    """
        Member functions of the user class for the eve arrival simulation
    """
    
    def __init__(self, a_j, x_j, alpha_j, r_j, xi_j=0.0):
        """
        This function initializes the user class.
        
        Parameters
        ----------
            a_j: list
                arrival time 
            x_j: list
                charging demand
            alpha_j:
                impatience factor
            r_j: 
                charging rates
            
        Returns
        -------
        n/a
        
        Notes
        -----
        n/a
 
        """
        self.a_j = a_j                          # Arrival time(h)
        self.x_j = x_j                          # Charging Demand (kWh)
        self.alpha_j = alpha_j                  # Impatience factor ($/hr.)
        self.r_j = r_j                          # Charging rate (kW)
        self.u_j = self.x_j/self.r_j            # Time to charge
        self.xi_j = xi_j 
        self.c_j = np.maximum(xi_j, self.x_j/self.r_j)
        self.finaltime = self.a_j + self.u_j    # Final time
        
    def get_user_attribute_table(self):
        """
            This function prints out a table of all the user parameters
        Parameters
        ----------
            n/a
  
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
 
        """
        t = PrettyTable(['User Attribute', 'Value'])
        t.add_row(['Arrival Time', self.a_j])
        t.add_row(['Charging Demand', self.x_j])
        t.add_row(['Impatience Factor', self.alpha_j])
        t.add_row(['Rate', self.r_j])
        t.add_row(['Time to Charge', self.u_j])
        print(t)

class UserCont:
    """
        Member functions of the user class for the eve arrival simulation
    """
    
    def __init__(self, a_j, x_j, alpha_j, xi_j, u_j):
        """
        This function initializes the user class.
        
        Parameters
        ----------
            a_j: list
                arrival time 
            x_j: list
                charging demand
            alpha_j:
                impatience factor
            r_j: 
                charging rates
            
        Returns
        -------
        n/a
        
        Notes
        -----
        n/a
 
        """
        self.a_j = a_j                          # Arrival time(h)
        self.x_j = x_j                          # Charging Demand (kWh)
        self.alpha_j = alpha_j                  # Impatience factor ($/hr.)
        self.r_j = self.x_j/u_j                          # Charging rate (kW)
        self.u_j = u_j           # Time to charge
        self.xi_j = xi_j 
        # self.c_j = self.u_j
        self.finaltime = self.a_j + self.u_j    # Final time

class ChargingFacilityDiscrete:
    """
        Member functions of the charging facility class. Contains functions & variables 
        for EV simulation.
    """
    
    def __init__(self, V, r, T, alpha_list, x_list, xi_list, lambdaval,  num_t_samples, model, Beta=0.0):
        """
            This function initializes the user class.
        
        Parameters
        ----------
            V : numpy array
                y-intercept of pricing function
            r : list
                charging rate (i.e., inverse of pricing function slope)
            T: float
                simulation total time (hrs.)
            alpha_list: list
                contains min alpha, max alpha, and alpha's distribution (impatience factor, $/hr)
            x_list: list
                contains min x, max x, and x's distribution (charging demand, kWh)
            xi_list: list
                contains min xi, max xi, and xi's distribution (time spent at charging facility, kWh)
            lambaval: list
                value of lambda 
            num_t_samples: list
                number of time samples
            model: string

                
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
             
        """
        self.model = model
        self.V_coeff = V
        self.r_coeff = r
        self.T = T
        self.Beta = Beta
        self.num_t_samples = num_t_samples
        self.L = len(self.r_coeff)

        # Pull min & max values of demand & impatience
        self.alpha_minmax = np.array(alpha_list[0:2])
        self.x_minmax = np.array(x_list[0:2])

        # Distributions Strings Stored
        self.alpha_distribution = alpha_list[2]
        self.x_distribution = x_list[2]

        # Considerations for special case
        if model != "specialcase":
            self.xi_minmax = np.array(xi_list[0:2])
            self.xi_distribution = xi_list[2]

        self.lambda_val = lambdaval

        # Compute Probability Function will be min()
        self._compute_probability_of_min()

        # Compute probability of being minimum with parking
        if self.model != "specialcase":
            self._compute_probability_of_min_parking()

        # Compute Expectation
        self.exp_xj = self._get_expectation_xj()
        self.exp_rj = self._get_expectation_rj()

        # Compute expected value of xij & cj (case applicable)
        if self.model != "specialcase":
            self.exp_xij = self._get_expectation_xij()
            self.exp_xjrj, self.exp_cj = self._get_expectation_numerical()
        else:
            self._get_expectation_rj()
            self.exp_cj = self._get_expectation_cj()

        self.exp_eta = lambdaval*self.exp_cj
        self.exp_eta_act = lambdaval*self.exp_xjrj

        self._get_expectation_second_moment_rj()
        
        # # Compute Upper Bounds
        self._compute_M_delta()
        self._compute_R_delta()

        # Catch usage of unsupported distributions for problem with parking
        if self.model != "specialcase":
            if self.x_distribution != "uniform" or self.xi_distribution != "uniform" or self.alpha_distribution != "uniform":
                    sys.exit("All distributions must be \"uniform\".") 


    def _get_price_func_param(self, x_j, alpha_j, xi_j = 0.0):
        """
            This function returns the rate of the function which minimizes the prices for a particular user.
        
        Parameters
        ----------
            x_j: numpy array
            
                user demand (kWh)
                
            alpha_j: numpy array
            
                user impatience factor ($/hr.)
            
        Returns
        -------
            self.r_coeff[indexofmin]: numpy array
                
                rate of cost function which minimizes user j's cost
        
        Notes
        -----
            n/a
         
        """
        store_price = []

        if self.model == "specialcase":
            for ii in range(0, len(self.V_coeff)):
                store_price.append(x_j*(self.V_coeff[ii] + alpha_j/self.r_coeff[ii]))
        else:
            for ii in range(0, len(self.V_coeff)):
                store_price.append(x_j*self.V_coeff[ii] + alpha_j*np.maximum(0.0, x_j/self.r_coeff[ii] - xi_j)
                    + self.Beta*np.maximum(0.0, xi_j - x_j/ self.r_coeff[ii]))
        # print(min(store_price))
        indexofmin = store_price.index(min(store_price))

        return self.r_coeff[indexofmin]
    
    def _get_expectation_cj(self):
        """
            This function computes the expectation of the random variable cj = max(xi_j, xj/rj).
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        
        if self.model == "specialcase":
            exp_cj = self._get_expectation_xjrj
        else:
            a = max(min(self.xi_minmax), min(self.x_minmax)/max(self.r_coeff))
            b = min(max(self.xi_minmax), max(self.x_minmax)/min(self.r_coeff))
            print("Lower Bound of Integration cj: ", min(self.xi_minmax), min(self.x_minmax)/max(self.r_coeff))
            print("Upper Bound of Integration cj: ", max(self.xi_minmax), max(self.x_minmax)/min(self.r_coeff))
            def C_first_moment(c):
                val = c*self._PDF_cj(c)
                # print(val)
                return val
            
            self.exp_cj = quad(C_first_moment, a,b)[0]

            
            return self.exp_cj

    def _PDF_cj(self, cj):
        """
            This function is the PDF of cj = max(xi_j, x_j/r_j).
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        pdf_cj = 0.0

        # Loop through definition of PDF of xj/rj
        for ii in range(0, self.L):
            if self.model != "specialcase":
                pdf_cj += self._PDF_cj_cond_R(cj, self.r_coeff[ii]) * self.price_func_min_probability[ii] 
            else:
                pdf_cj += self._PDF_cj_cond_R(cj, self.r_coeff[ii]) * self.price_func_min_probability[ii] 
        

        return pdf_cj

    def _PDF_cj_cond_R(self, cj, R):
        """
            This function is the PDF of cj = max(xi_j, x_j/r_j).
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """

        return self._PDF_xij(cj)*self._CDF_xj(cj*R)*R + R*self._PDF_xj(cj*R)*self._CDF_xij(cj) 

    def _PDF_alphaj(self, alphaj):
        """
            This function is the PDF of xj.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        if self.alpha_distribution == "uniform":
            alphamin = min(self.alpha_minmax)
            alphamax = max(self.alpha_minmax)

            pdf_alpha = 1/(alphamax - alphamin)  * (alphamin < alphaj) * (alphaj < alphamax)

        return pdf_alpha

    def _CDF_xj(self, xj):
        """
            This function computes the CDF of x_j
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """

        cdf_xj = quad(self._PDF_xj, 0, xj)[0]

        return cdf_xj

    def _PDF_xj(self, xj):
        """
            This function is the PDF of xj.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        if self.x_distribution == "uniform":
            xmin = min(self.x_minmax)
            xmax = max(self.x_minmax)

            pdf_X = 1/(xmax - xmin)  * (xmin < xj) * (xj < xmax)

        return pdf_X 
    
    def _CDF_xij(self, xij):
        """
            This function computes the PDF of x_j/r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """

        cdf_xij = quad(self._PDF_xij, 0, xij)[0]

        return cdf_xij

    def _PDF_xij(self, xij):
        """
            This function is the PDF of xj.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        if self.xi_distribution == "uniform":
            ximin = min(self.xi_minmax)
            ximax = max(self.xi_minmax)

            pdf_Xi = 1/(ximax - ximin) * (ximin < xij) * (xij < ximax)

        return pdf_Xi 

    def _CDF_xjrj(self, xjrj):
        """
            This function computes the PDF of x_j/r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """

        cdf_xjrj = quad(self._PDF_xjrj, 0, xjrj)[0]

        return cdf_xjrj


    def _PDF_xjrj(self, xjrj):
        """
            This function computes the PDF of x_j/r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        # pairs = self._compute_index_pairs()
        # pairs = np.concatenate(([[-999,0]], pairs))
        # pairs = np.concatenate((pairs, [[self.L - 1, 999]]))
        # self._PDF_xj_cond_R_alpha_xi(1,1,1,1,pairs[0])


        pdf_xjrj = 0.0

        # Loop through definition of PDF of xj/rj
        for ii in range(0, self.L):
            if self.model != "specialcase":
                ### TODO: Add nonspecial case implementation 
                pdf_xjrj += self.r_coeff[ii]* self._PDF_xj_cond_R(xjrj * self.r_coeff[ii], self.r_coeff[ii], ii) * self.price_func_min_probability[ii] 
                print(pdf_xjrj)
            else:
                pdf_xjrj += self.r_coeff[ii]* self._PDF_xj(xjrj * self.r_coeff[ii]) * self.price_func_min_probability[ii] 


        return pdf_xjrj

    def _PDF_xj_cond_R(self, xj, R, idx):
        """
            This function computes the PDF of x_j/r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        def total_prob_cont(alphaj, xij, R, xj):
            val = self._PDF_xj_cond_R_alpha_xi(xj, alphaj,xij, R, idx)*self._PDF_xij(xij)*self._PDF_alphaj(alphaj)
            return val

        return dblquad(total_prob_cont, min(self.alpha_minmax), max(self.alpha_minmax), min(self.xi_minmax), max(self.xi_minmax), args=(R, xj))[0]

    def _PDF_xj_cond_R_alpha_xi(self, xj, alphaj, xij, R, idx):
        """
            This function computes the PDF of x_j/r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        if idx == 0:

            return R * self._PDF_xj(R * xj)  \
                    * (xj < (1/R) * xij * (alphaj + self.Beta)/ ((self.V_coeff[idx] - self.V_coeff[idx + 1]) + (alphaj/self.r_coeff[idx] + self.Beta/self.r_coeff[idx + 1])) ) 
        elif idx == self.L - 1:

            return R * self._PDF_xj(R * xj) \
                    * ((1/R) * xij * (alphaj + self.Beta)/((self.V_coeff[idx - 1] - self.V_coeff[idx]) + (alphaj/self.r_coeff[idx - 1] + self.Beta/self.r_coeff[idx])) < xj) 
        else:

            return R * self._PDF_xj(R * xj) * ((1/R) * xij * (alphaj + self.Beta)/((self.V_coeff[idx - 1] - self.V_coeff[idx]) + (alphaj/self.r_coeff[idx - 1] + self.Beta/self.r_coeff[idx ])) < xj) \
                    * (xj < (1/R) * xij * (alphaj + self.Beta)/ ((self.V_coeff[idx] - self.V_coeff[idx + 1]) + (alphaj/self.r_coeff[idx] + self.Beta/self.r_coeff[idx + 1])) ) 
        # print(pair)        
        # return R * _PDF_xj(R * xj) * 1

    def _get_expectation_xj(self):
        """
            This function computes the expectation .
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
         

        def X_first_moment(X):
            val = X*self._PDF_xj(X)
            return val
        
        self.exp_xj = quad(X_first_moment, min(self.x_minmax), max(self.x_minmax))[0]
        
        return self.exp_xj

    def _get_expectation_xij(self):
        """
            This function computes the expectation .
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """

        def Xi_first_moment(Xi):
            val = Xi*self._PDF_xij(Xi)
            return val
        
        self.exp_xij = quad(Xi_first_moment, min(self.xi_minmax), max(self.xi_minmax))[0]
        
        return self.exp_xij

    def _get_expectation_rj(self):
        """
            This computes the expect value of the charging rate, r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        r_index = 0
        self.exp_rj = 0
        for probability in self.price_func_min_probability:
            self.exp_rj = self.exp_rj + probability*self.r_coeff[r_index]
            r_index = r_index + 1
             
        return self.exp_rj

    def _get_expectation_rj_inv(self):
        """
            This computes the expect value of the charging rate, r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        r_index = 0
        self.exp_rj = 0
        for probability in self.price_func_min_probability:
            self.exp_rj = self.exp_rj + probability*(1/self.r_coeff[r_index])
            r_index = r_index + 1
             
        return self.exp_rj_inv

    def _get_expectation_xjrj(self):
        """
            Computes the expected value of the time to charge.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            exp_xj/exp_j: numpy array
                
                expectation of u, i.e., expectation of time to charge
        
        Notes
        -----
            n/a
         """

        if self.model == "specialcase":
            self.exp_xj = self._get_expectation_xj()
            self.exp_rj_inv = self._get_expectation_rj_inv()
            
            self.exp_xjrj = self.exp_xj*self.exp_rj_inv
        else: 
            def XjRj_first_moment(xjrj):
                val = xjrj*self._PDF_xjrj(xjrj)
                return val

            self.exp_xjrj = quad(XjRj_first_moment, 0, max(self.x_minmax)/min(self.r_coeff))[0]
            # self.exp_xjrj = quad(XjRj_first_moment, 0, np.infty)[0]

            print("Integration", self.exp_xjrj)

        return self.exp_xjrj

    def _get_expectation_numerical(self):
        """
            Computes the expected value of the time to charge numerically
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            exp_xj/exp_j: numpy array
                
                expectation of u, i.e., expectation of time to charge
        
        Notes
        -----
            n/a
         """
        elements = 1000
        samples = 10
        exp_xjrj = []
        exp_cj = []
        for jj in range(0, samples):
            rj_array = np.empty([])
            xj_array = np.random.uniform(min(self.x_minmax), max(self.x_minmax), elements)
            xij_array = np.random.uniform(min(self.xi_minmax), max(self.xi_minmax), elements)
            alphaj_array = np.random.uniform(min(self.alpha_minmax), max(self.alpha_minmax), elements)

            for ii in range(0, elements):
                if self.model != "specialcase":
                    rj_array = np.append(rj_array, self._get_price_func_param(xj_array[ii], alphaj_array[ii], xij_array[ii]))
                else:
                    rj_array = np.append(rj_array, self._get_price_func_param(xj_array[ii], alphaj_array[ii]))

            rj_array = np.delete(rj_array,0)

            xjrj_array = np.divide(xj_array, rj_array)
            cj_array = np.maximum(xij_array, np.divide(xj_array, rj_array))

            exp_xjrj.append(np.mean(np.divide(xj_array, rj_array)))
            exp_cj.append(np.mean(cj_array))

        self.exp_xjrj = np.mean(np.array(exp_xjrj))
        self.exp_cj = np.mean(np.array(exp_cj))

        return self.exp_xjrj, self.exp_cj

    def _get_expectation_second_moment_rj(self):
        """
            This computes the second moment of the charging rate, r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        r_index = 0
        self.exp_second_moment_rj = 0
        for probability in self.price_func_min_probability:
            self.exp_second_moment_rj = self.exp_second_moment_rj + probability*self.r_coeff[r_index]**2
            r_index = r_index + 1
                 
    def _compute_probability_of_min(self):
        """
            This function computes the probability of a function being the min
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        # Initialize delta arrays
        delta_Vcoeff = np.empty([np.size(self.V_coeff), np.size(self.V_coeff)])
        delta_rcoeff = np.empty([np.size(self.r_coeff), np.size(self.r_coeff)])
        
        # Compute the delta's of P_ki and r_ki 
        for ii in range(0,np.size(self.V_coeff)):
            for jj in range(0,np.size(self.V_coeff)):
                    delta_Vcoeff[ii][jj] = self.V_coeff[ii] - self.V_coeff[jj]
                    delta_rcoeff[ii][jj] = 1/self.r_coeff[ii] - 1/self.r_coeff[jj]
        
        delta_fraction = np.empty([np.size(self.V_coeff), np.size(self.V_coeff)])
        
        # Compute the delta's of V_ki and r_ki 
        for ii in range(0,np.size(self.V_coeff)):
            for jj in range(0,np.size(self.V_coeff)):
                    delta_fraction[ii][jj] = delta_Vcoeff[ii][jj]/delta_rcoeff[jj][ii]

        self.delta_fraction = delta_fraction
        # 
        if self.alpha_distribution == 'uniform':
            def pdf_A(alpha, a, b):
                val = 1/(b - a)
                return val
                
        # Place min & max alpha values in numpy array        
        alphamin = min(self.alpha_minmax)
        alphamax = max(self.alpha_minmax)
        
        # Define bounds of uniform distribution
        a = alphamin
        b = alphamax
        
        self.price_func_min_probability = np.empty(self.V_coeff.size)
        lowerbound = np.empty(self.V_coeff.size)
        upperbound = np.empty(self.V_coeff.size)
        
        # Loop through all N pricing functions
        for ii in range(0, np.size(self.V_coeff)):
            lower_indices = np.arange(0, ii) # Define indices less than k
            
            if ii != np.size(self.V_coeff) - 1:
                upper_indices = np.arange(ii+1, self.V_coeff.size) # Define indices greater than k
            else:
                upper_indices = np.arange(ii, self.V_coeff.size) # Define indices greater than k

            
            lower_delta_fraction = np.take(delta_fraction[ii], lower_indices)
            upper_delta_fraction = np.take(delta_fraction[ii], upper_indices)

            lower_delta_fraction = lower_delta_fraction[lower_delta_fraction >= 0]
            upper_delta_fraction = upper_delta_fraction[upper_delta_fraction >= 0]
            
            if lower_delta_fraction.size !=0:
                lowerbound[ii] = max(alphamin, np.max(lower_delta_fraction))
            else:
                lowerbound[ii] = alphamin
            
            # Define upper bound of integration
            # if-statement to catch empty "upper_delta_fraction" (i.e., when k = N)
            if upper_delta_fraction.size != 0:
                upperbound[ii] = min(alphamax, np.min(upper_delta_fraction))
            else:
                upperbound[ii] = alphamax

            if self.alpha_distribution == 'uniform':
                self.price_func_min_probability[ii] = max(0.0, quad(pdf_A, lowerbound[ii], upperbound[ii], args=(a,b))[0])
            elif self.alpha_distribution == 'trunc_normal':
                # Lower & upper limits of truncated normal distribution
                lower = min(self.alpha_minmax)
                upper = max(self.alpha_minmax)
                mu = (upper + lower)/2      # Mean
                sigma = (upper - mu)*.5     # Standard deviation
                self.price_func_min_probability[ii] = max(0.0, \
                    stats.truncnorm.cdf((upperbound[ii] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma)-\
                    stats.truncnorm.cdf((lowerbound[ii] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma))
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def _compute_probability_of_min_parking(self):
        """
            This function computes the probability of a function being the min when 
            pricing function charge parking (Automatica 2020).
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        print_k = -1
        # print("R^L < rho_j \n")
        ################ CASE 1 (R^L < rho) ########################

        self.price_func_min_probability_case1 = np.zeros_like(self.price_func_min_probability)
        # Compute first portion of min probability for two types of distributions
        if self.alpha_distribution == 'uniform':            
            # Define Uniform distribution 
            def rho_distribution(rho, x_min, x_max, xi_min, xi_max):
                if ii == print_k:
                    print("k=" + str(ii + 1), rho, (np.minimum(xi_max, x_max/rho)**2 \
                            - np.maximum(xi_min, x_min/rho)**2)/(2 * (xi_max - xi_min) * (x_max - x_min)))
                return (np.minimum(xi_max, x_max/rho)**2 \
                        - np.maximum(xi_min, x_min/rho)**2)/(2 * (xi_max - xi_min) * (x_max - x_min))

            # # Plot ratio PDF
            # rho_domain = np.linspace(.0001,1000, 1000)
            # vals = rho_distribution(rho_domain, min(self.x_minmax), max(self.x_minmax), min(self.xi_minmax), max(self.xi_minmax))
            # # print(vals)
            # fig = plt.figure()
            # ax = plt.subplot(111)
            # ax.plot(rho_domain, vals)
            # ax.grid(True)
            # ax.set_xlabel(r'$\rho$')
            # ax.set_ylabel(r'$f_{\rho} }$')
            # ax.set_title(r'PDF of $f_P$')     

            # print(self.L, len(self.r_coeff))

            # Compute rho integral from R^L < rho < np.inf 
            for ii in range(0, len(self.price_func_min_probability)):
                int_val, _ = quad(rho_distribution, self.r_coeff[self.L - 1], np.inf,\
                                args=(min(self.x_minmax) , max(self.x_minmax) , min(self.xi_minmax) , max(self.xi_minmax)))

                self.price_func_min_probability_case1[ii] = int_val*self.price_func_min_probability[ii]
#            print(self.price_func_min_probability_case1)

        elif self.alpha_distribution == 'trunc_normal':
            pass     
        # print("\n rho_j < R^1")
        ################ CASE 2 (rho < R^1 ) ########################

        # Initialize delta arrays
        delta_Vcoeff = np.empty([np.size(self.V_coeff), np.size(self.V_coeff)])
        delta_rcoeff = np.empty([np.size(self.r_coeff), np.size(self.r_coeff)])
        
        # Compute the delta's of P_ki and r_ki 
        for ii in range(0,np.size(self.V_coeff)):
            for jj in range(0,np.size(self.V_coeff)):
                    delta_Vcoeff[ii][jj] = self.V_coeff[ii] - self.V_coeff[jj]
                    delta_rcoeff[ii][jj] = 1/self.r_coeff[ii] - 1/self.r_coeff[jj]

        coeff = np.add(delta_Vcoeff, - self.Beta*delta_rcoeff)
        coeff = np.sum(coeff < 0, axis=1)
        coeff = coeff == (len(self.V_coeff) - 1)
        coeff = coeff * 1 # convert bool to number

        self.price_func_min_probability_case2 = np.array(coeff, dtype=float)

        if self.alpha_distribution == 'uniform':            
            # Compute rho integral from 0 < rho < R^1
            for ii in range(0, len(self.price_func_min_probability_case2)):
                int_val, _ = quad(rho_distribution, 0, self.r_coeff[0],\
                                args=(min(self.x_minmax) , max(self.x_minmax) , min(self.xi_minmax) , max(self.xi_minmax)))

                self.price_func_min_probability_case2[ii] = int_val * self.price_func_min_probability_case2[ii]

        elif self.alpha_distribution == 'trunc_normal':
            pass    

        ################ CASE 3 (R^m < rho < R^(m+1) ) ########################
        # Compute bin pairs, i.e., (m, m+1)
        index_pairs = self._compute_index_pairs()
        self.price_func_min_probability_case3 = np.zeros_like(self.r_coeff, dtype=float)
#        print(self.delta_fraction)
        # Loop through bins
        for pair in index_pairs:
            m = pair[0]
            m_plus_one = pair[1]
#            print("\n R Value Range", self.r_coeff[m], "to", self.r_coeff[m_plus_one])
#            print("\n Bin Indices", m, "to", m_plus_one, "\n")
            # print("Partition \n m: ",m+1, "m+1: ",m_plus_one+1)
            # Loop through k-values possible in each bin
            for k in range(0, self.L):
#                print("Current K-value: ", k)
                # Case when k is less than or equal to m
                if k <= m:
                    # Impatience factor distribution is uniform
                    if self.alpha_distribution == 'uniform':  
                        def rho_distribution_subcase_A(rho, x_min, x_max, xi_min, xi_max):
                            # Define PDF of rho 
                            rho_pdf = (np.minimum(xi_max, x_max/rho)**2 \
                                - np.maximum(xi_min, x_min/rho)**2)/(2 * (xi_max - xi_min) * (x_max - x_min))
                            
                            # Upper bound on Alpha Integration
                            if k == m:
                                # Upper bound on integration over alpha
                                upper = np.minimum(max(self.alpha_minmax), \
                                        (self.Beta*(1/rho - 1/self.r_coeff[m_plus_one]) - (self.V_coeff[k] - self.V_coeff[m_plus_one]) )/ (1/self.r_coeff[k] - 1/rho) )
                            else:
                                # Upper bound on integration over alpha
                                upper = np.minimum(max(self.alpha_minmax), \
                                    (self.Beta*(1/rho - 1/self.r_coeff[m_plus_one])- (self.V_coeff[k] - self.V_coeff[m+1]) )/ (1/self.r_coeff[k] - 1/rho) )
                                # min function used twice since can 
                                upper = np.minimum(upper,-(self.V_coeff[k] - self.V_coeff[k+1])/(1/self.r_coeff[k] - 1/self.r_coeff[k+1]))

                            # Lower bound of alpha integral 
                            if k > 0:    
                                lower = np.maximum(min(self.alpha_minmax), \
                                        -(self.V_coeff[k] - self.V_coeff[k-1] )/(1/self.r_coeff[k] - 1/self.r_coeff[k-1]) )
                            else:
                                lower = min(self.alpha_minmax)

                            prob_given_rho = (upper - lower ) * 1/(max(self.alpha_minmax) - min(self.alpha_minmax))
                                             
                            # print("Rho Value: ", rho, "Integral Value: ", \
                            #       rho_pdf*prob_given_rho, "Value of Rho PDF", rho_pdf,\
                            #       "Value of alpha integral", prob_given_rho)
                            if k == print_k:
                                print("k=" + str(k+1), rho, np.maximum(0,prob_given_rho))

                            return rho_pdf*np.maximum(0,prob_given_rho)

                        int_val, _ = quad(rho_distribution_subcase_A, self.r_coeff[m], self.r_coeff[m_plus_one],\
                                args=(min(self.x_minmax) , max(self.x_minmax) , min(self.xi_minmax) , max(self.xi_minmax)))
                        
                        self.price_func_min_probability_case3[k] += int_val
#                        print("Probabilities", self.price_func_min_probability_case3)

                    elif self.alpha_distribution == 'trunc_normal':  
                        pass
                # Case when k is greater than m+1
                elif k > m_plus_one:
                    self.price_func_min_probability_case3[k] += 0.0
#                    print("Probabilities", self.price_func_min_probability_case3)
                
                # Case when k is equal to than m+1
                elif k == m_plus_one:
                    if self.alpha_distribution == 'uniform':  
                        def rho_distribution_subcase_C(rho, x_min, x_max, xi_min, xi_max):
                            # Define PDF of rho 
                            rho_pdf = (np.minimum(xi_max, x_max/rho)**2 \
                                - np.maximum(xi_min, x_min/rho)**2)/(2 * (xi_max - xi_min) * (x_max - x_min))
                            
                            # Upper bound of alpha integral
                            upper = max(self.alpha_minmax)

                            # Lower bound of alpha integral
                            lower = np.maximum(min(self.alpha_minmax), (self.Beta*(1/self.r_coeff[k] - 1/rho)\
                                                - (self.V_coeff[k] - self.V_coeff[m]))/(1/rho - 1/self.r_coeff[m]))


                            prob_given_rho = (upper - lower) * 1/(max(self.alpha_minmax) - min(self.alpha_minmax))
                                             
                            # print("Rho Value: ", rho, "Integral Value: ", \
                            #           rho_pdf*prob_given_rho, "Value of Rho PDF", rho_pdf,\
                            #           "Value of alpha integral", prob_given_rho)
                            if k == print_k:
                                print("k="+ str(k+1), rho, np.maximum(0,prob_given_rho))
                            return  rho_pdf*np.maximum(0,prob_given_rho)

                        int_val, _ = quad(rho_distribution_subcase_C, self.r_coeff[m], self.r_coeff[m_plus_one],\
                                args=(min(self.x_minmax) , max(self.x_minmax) , min(self.xi_minmax) , max(self.xi_minmax)))

                            
                        self.price_func_min_probability_case3[k] += int_val
#                        print("Probabilities", self.price_func_min_probability_case3)
                        
                    elif self.alpha_distribution == 'trunc_normal':  
                        pass

        self.price_func_min_probability =   self.price_func_min_probability_case1 + \
                                            self.price_func_min_probability_case2 + \
                                            self.price_func_min_probability_case3
    def _compute_index_pairs(self):
        """
            This function computes the index pairs for R^m < rho < R^(m+1)
            i.e. compute (m, m+1) pairs. 
            
        Parameters
        ----------
            n/a
            
        Returns
        -------
            Index pairs for bins 
        
        Notes
        -----
            n/a
        """
        index_pairs = []
        for ii in range(0, self.L):
            if ii is not (self.L - 1):
                index_pairs.append([ii, ii+1])
        
        return np.array(index_pairs)
    
    def _compute_M_delta(self):
        """
            This function computes the probability of a function being the min
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
#        self.delta_M_range = np.linspace(1e-2, 1, 100)
#        self.M_func = self.exp_eta + 1/3 * np.log(np.divide(1, self.delta_M_range)) + \
#        np.sqrt(1/9 * np.log(np.divide(1, self.delta_M_range))**2 + \
#        2*self.exp_eta*np.log(np.divide(1, self.delta_M_range)))
#        
#        self.M_func_loose = self.exp_eta + 2/3 * np.log(np.divide(1, self.delta_M_range)) + \
#        np.sqrt(2*self.exp_eta*np.log(np.divide(1, self.delta_M_range)))
        
        self.confidence_delta_M = np.empty([])
        self.script_M = np.linspace(.05, 100, 1000) + self.exp_eta
#        temp_delta_M = 0.0
#        for M_val in self.script_M:s
        numerator = -(self.script_M - self.exp_eta)**2
        denominator = 2*(self.exp_eta + 1/3 * (self.script_M - self.exp_eta))
        
        self.confidence_delta_M  = np.exp(np.divide(numerator, denominator))

    def _compute_R_delta(self):
        """
            This function computes the probability of a function being the min
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """ 
        self.script_R = np.linspace(self.exp_eta_act*self.exp_rj, 2300,20)
  
        self.confidence_delta_R = np.empty([])

        for script_R_values in self.script_R:
            sum_temp_var = 0.0 # temporary variable to perform summation
            
            # Define lower and upper bound of summation
            M_upper = int(np.floor(script_R_values/self.exp_rj))
            M_lower = int(np.ceil(script_R_values/max(self.r_coeff)))
#            M_lower = 0
            
            # Define numerator of and denominator of exponential
            numerator = -(M_upper - self.exp_eta_act)**2     
            denominator = 2*(self.exp_eta_act + 1/3 * (M_upper - self.exp_eta_act))
            
            # Define m values over which to perform sum
            m = np.linspace(M_lower, M_upper, M_upper - M_lower + 1)
            
            exp_numerator = -(script_R_values - m*self.exp_rj)**2
            exp_denominator = 2*(m*self.exp_second_moment_rj + (1/3)*max(self.r_coeff)*(script_R_values - m*self.exp_rj))
            sum_total_probability = np.multiply(np.exp(np.divide(exp_numerator, exp_denominator)), \
                                    stats.poisson.pmf(m, self.exp_eta_act, loc = 0))
            
            if script_R_values <= self.exp_eta_act*self.exp_rj:
                sum_temp_var = 1
            else:
                sum_temp_var = np.sum(sum_total_probability) +  np.exp(np.divide(numerator, denominator))
                sum_temp_var= min(1, sum_temp_var)


#            print(sum_temp_var)
            self.confidence_delta_R = np.append(self.confidence_delta_R, sum_temp_var)
#        self.script_R = self.script_R + self.exp_eta*self.exp_rj
        self.confidence_delta_R = np.delete(self.confidence_delta_R, [0], axis=0)
    def compute_sanity_check(self, iterations=1000):
        """
            This function computes a sanity check on expectation variables
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """ 
        # Initialize arrays
        X_array = np.empty([])
        Alpha_array = np.empty([])
        if self.model != "specialcase":
            Xi_array = np.empty([])
        R_array = np.empty([])

        for ii in range(0,iterations):
            if self.x_distribution == 'uniform':
                user_xj = np.random.uniform(min(self.x_minmax), max(self.x_minmax))
                
            # Draw impatience factor
            if self.alpha_distribution == 'uniform':
                user_alphaj = np.random.uniform(min(self.alpha_minmax), max(self.alpha_minmax))

                
            # Draw time spend at charging location (only if not special case model)
            if self.model != "specialcase":
                if self.xi_distribution == 'uniform':
                    user_xij = np.random.uniform(min(self.xi_minmax), max(self.xi_minmax))

            X_array = np.append(X_array, user_xj)
            Alpha_array = np.append(Alpha_array, user_alphaj)

            if self.model != "specialcase":
                Xi_array = np.append(Xi_array, user_xij)
                R_array = np.append(R_array, self._get_price_func_param(user_xj, user_alphaj, user_xij))
            else:
                R_array = np.append(R_array, self._get_price_func_param(user_xj, user_alphaj))

        # Arrays
        X_array = np.delete(X_array, 0)
        Alpha_array = np.delete(Alpha_array, 0)
        if self.model != "specialcase":
            Xi_array = np.delete(Xi_array, 0)
        R_array = np.delete(R_array, 0)
        # import pdb; pdb.set_trace()

        # Simulation-based Expectations
        print("\nSimulation-based Values")
        print("E[x] =", np.mean(X_array))
        if self.model != "specialcase":
            print("E[xi] =", np.mean(Xi_array))
            print("E[c] =", np.mean(np.maximum(Xi_array, np.divide(X_array, R_array))) )
        else:
            print("E[c] =", np.mean(np.mean(np.divide(X_array, R_array))))
        print("E[r] =", np.mean(R_array) )
        print("E[x/r] = ", np.mean(np.divide(X_array, R_array)))

        # PDF based bounds
        # Simulation-based Expectations
        print("\nIntegration based Values")
        print("E[x] =", self.exp_xj)
        if self.model != "specialcase":
            print("E[xi] =", self.exp_xij)
        print("E[c] =", self.exp_cj)
        print("E[r] =", self.exp_rj)
        print("E[x/r] = ", self.exp_xjrj)

    def get_pricingfunc_plot(self, DistFlag=None):
        """
            This function prints out all the pricing functions and saves them.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         
        """
        
        if DistFlag is None:
            alpha_range = np.linspace(min(self.alpha_minmax), max(self.alpha_minmax), 50)
            f = plt.figure()
            for ii in range(0,self.V_coeff.size):
                plt.plot(alpha_range, self.V_coeff[ii] + alpha_range/self.r_coeff[ii], label = r'$r_' + str(ii + 1) + ' ,V_' + str(ii + 1) + '$' )
                plt.xlabel(r"$\alpha$")
                plt.ylabel(r"$g_\ell (\cdot, \alpha)$")
                plt.grid(True, which='both')
            
            plt.legend()
            plt.show()
            f.savefig("./plots/pricingfunc.pdf", bbox_inches='tight')
            
        elif DistFlag == 'one':
            alpha_range = np.linspace(min(self.alpha_minmax), max(self.alpha_minmax), 50)
            fig1, ax1 = plt.subplots()
            for ii in range(0,self.V_coeff.size):
                ax1.plot(alpha_range, self.V_coeff[ii] + alpha_range/self.r_coeff[ii], label = r'$r_' + str(ii + 1) + ' ,V_' + str(ii + 1) + '$' )
                ax1.set_xlabel(r"$\alpha$")
                ax1.set_ylabel(r"$g_i(\cdot, \alpha)$")
                plt.grid(True, which='both')
                plt.legend()
            ax1.set_ylim([min(.95*self.V_coeff), 7.5])
            
            if self.alpha_distribution == 'trunc_normal':
                ax2 = ax1.twinx()
                
                # Lower & upper limits of truncated normal distribution
                lower = min(self.alpha_minmax)
                upper = max(self.alpha_minmax)
                mu = (upper + lower)/2      # Mean
                sigma = (upper - mu)*.5     # Standard deviation
                
                # Standardize Normal Distribution and get pdf values & plot
                alpharange_pdf = np.linspace(stats.truncnorm.ppf(0.00, (lower - mu) / sigma , (upper - mu) / sigma), stats.truncnorm.ppf(1.00, (lower - mu) / sigma , (upper - mu) / sigma), 100)
                ax2.plot(sigma*(alpharange_pdf  + mu/sigma), stats.truncnorm.pdf(alpharange_pdf , (lower - mu) / sigma , (upper - mu) / sigma), 'r--', lw=1.5, alpha=0.6, label='Trunc. Norm PDF')
                ax2.set_ylim([0, 1])
                ax2.set_ylabel(r'Probability')
                
                # Get percentile of normal distribution of region to shade
                get_min_alpha_percentile = stats.truncnorm.cdf((self.lowerbound[1] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma)
                get_max_alpha_percentile = stats.truncnorm.cdf((self.upperbound[1] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma)
                
                plt.legend()
                plt.show()
                fig1.savefig("./plots/pricingfunc_with_dist.pdf", bbox_inches='tight')
                
        elif DistFlag == 'two':
                chosen_ii = 1
                alpha_range = np.linspace(min(self.alpha_minmax), max(self.alpha_minmax), 50)
                fig1 = plt.figure()
                fig1.subplots_adjust(wspace=.1)
                ax1 = fig1.add_subplot(1, 2, 1)
                for ii in range(0,self.V_coeff.size):
                    ax1.plot(alpha_range, self.V_coeff[ii] + alpha_range/self.r_coeff[ii], label = r'$R^' + str(ii + 1) + ' ,V^' + str(ii + 1) + '$' )
                    ax1.set_xlabel(r"$\alpha$")
                    ax1.set_ylabel(r"$g_\ell(x, \cdot)$")
                    plt.grid(True, which='both')
                    plt.legend()
                ax1.set_ylim([min(.95*self.V_coeff), 7.5])
                
                if self.alpha_distribution == 'trunc_normal':
                    ax2 = ax1.twinx()
                    
                    # Lower & upper limits of truncated normal distribution
                    lower = min(self.alpha_minmax)
                    upper = max(self.alpha_minmax)
                    mu = (upper + lower)/2      # Mean
                    sigma = (upper - mu)*.5     # Standard deviation
                    
                    # Standardize Normal Distribution and get pdf values & plot
                    alpharange_pdf = np.linspace(stats.truncnorm.ppf(0.00, (lower - mu) / sigma , (upper - mu) / sigma), stats.truncnorm.ppf(1.00, (lower - mu) / sigma , (upper - mu) / sigma), 100)
                    ax2.plot(sigma*(alpharange_pdf  + mu/sigma), stats.truncnorm.pdf(alpharange_pdf , (lower - mu) / sigma , (upper - mu) / sigma), 'r--', lw=1.5, alpha=0.6)
                    ax2.set_ylim([0, 1])
                    ax2.axis('off')
                
                ax3 = fig1.add_subplot(1, 2, 2)
                # Plot
                for ii in range(0,self.V_coeff.size):
                    if ii != chosen_ii:
                        ax3.plot(alpha_range, self.V_coeff[ii] + alpha_range/self.r_coeff[ii],\
                                 color='gray', alpha=.3)
                    else:
                        ax3.plot(alpha_range, self.V_coeff[ii] + alpha_range/self.r_coeff[ii], color='C1')
                ax3.set_xlabel(r'$\alpha$')
                ax3.grid(True)
                ax3.set_ylim([min(.95*self.V_coeff), 7.5])
                ax3.tick_params(axis='y', which='major',labelleft=False)
                
                if self.alpha_distribution == 'trunc_normal':
                        ax4 = ax3.twinx()
                        
                        # Standardize Normal Distribution and get pdf values & plot
                        alpharange_pdf = np.linspace(stats.truncnorm.ppf(0.00, (lower - mu) / sigma , (upper - mu) / sigma), stats.truncnorm.ppf(1.00, (lower - mu) / sigma , (upper - mu) / sigma), 100)
                        ax4.plot(sigma*(alpharange_pdf  + mu/sigma), stats.truncnorm.pdf(alpharange_pdf , (lower - mu) / sigma , (upper - mu) / sigma), 'r--', lw=1.5, alpha=0.6, label=r'$f_A(\alpha)$')
                        ax4.set_ylim([0, 1])
                        ax4.set_ylabel(r'Probability')
                        
                        # Get percentile of normal distribution of region to shade
                        get_min_alpha_percentile = stats.truncnorm.cdf((self.lowerbound[chosen_ii] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma)
                        get_max_alpha_percentile = stats.truncnorm.cdf((self.upperbound[chosen_ii] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma)
                        
                        # Create range for region to shade
                        alpharange_pdf_shade = np.linspace(stats.truncnorm.ppf(\
                                        get_min_alpha_percentile, (lower - mu) / sigma ,\
                                        (upper - mu) / sigma), stats.truncnorm.ppf(get_max_alpha_percentile,\
                                        (lower - mu) / sigma , (upper - mu) / sigma), 100)
                        # Fill probability Distribution between integration bounds
                        plt.fill_between(sigma*(alpharange_pdf_shade + mu/sigma),\
                                         stats.truncnorm.pdf(alpharange_pdf_shade, (lower - mu) / sigma ,\
                                        (upper - mu) / sigma), color='r', alpha=.1)

                        plt.legend()
                        plt.show()
                fig1.savefig("./plots/pricingfunc_with_dist_twoplot.pdf", bbox_inches='tight')

    def get_sim_plots(self):
        """
            This function runs the charging facility simulation.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
         
        
        f = plt.figure()
        plt.plot(self.time_for_plot,self.activeusers)
        plt.grid('True')
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\eta(t)$")
        plt.title(r"Number of users versus Time (hr.)")
        plt.show()
        f.savefig("./plots/num_of_users_vs_t.pdf", bbox_inches='tight')
        
        f = plt.figure()
        plt.plot(self.time_for_plot, self.totalrate)
        plt.grid('True')
        plt.xlabel(r"$t$")
        plt.ylabel(r"$R(t)$")
        plt.title(r"Charging Rate (kW) versus Time (hr.)")
        plt.show()
        f.savefig("./plots/chargerate_vs_t.pdf", bbox_inches='tight')
        
    def get_upperbound_plot(self, N_percentile_store=None, percentiles=None, R_percentile_store=None, percentiles2=None):
        """
            Plots the results from the 
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        if N_percentile_store is None:
            fig = plt.figure(0)
            plt.plot(1 - self.delta_M_range, self.M_func)
            plt.plot(1 - self.delta_M_range, self.M_func_loose)
            plt.grid('True')
            # plt.xlabel(r'1 - $\delta(\mathcal{M})$')
            plt.xlabel(r'Number of Present Users, $\mathcal{M}$')
            plt.ylabel(r"$\mathbb{P}\left(\eta(t) < \mathcal{M}\right)$")
            plt.ylim([0, 1.05])
            plt.show()
        else:
            spacing = 5
            N_percentile_store = np.delete(N_percentile_store, [0], axis=0)
            N_percentile_means = np.mean(N_percentile_store, axis=0)

            f = plt.figure(0)
            plt.plot(self.script_M, 1- self.confidence_delta_M, label=r'Theoretical Bound', color=(74.0/255.0, 176.0/255.0, 166.0/255))
            plt.plot(N_percentile_means, percentiles/100, label=r'$\eta(t)$ Monte Carlo',color=(93.0/255.0, 58.0/255.0, 155.0/255))

            N_percentile_store = N_percentile_store[:, np.arange(1,97, spacing)-1]
            percentiles = np.arange(1,97,spacing)
            N_percentile_means = np.mean(N_percentile_store, axis=0)
            N_percentile_std = np.std(N_percentile_store, axis=0)
            # xerr = [N_percentile_store.min(axis=0)-N_percentile_means, N_percentile_means-N_percentile_store.max(axis=0)]
            xerr = [2*N_percentile_std, 2*N_percentile_std]
            print(xerr)
            plt.errorbar(N_percentile_means, percentiles/100, xerr=xerr, color=(93.0/255.0, 58.0/255.0, 155.0/255),fmt='none')
            plt.grid('True')
            plt.xlabel(r'Number of Present Users, $\mathcal{M}$')
            plt.ylabel(r"$\mathbb{P}\left(\eta(t) < \mathcal{M}\right)$")
            plt.xlim([0, max(self.script_M)])
            plt.ylim([0, 1.05])
            # plt.title(r'Total Active Users vs. Confidence Interval')
            plt.legend()
            plt.rcParams.update({'font.size': 14})    
            plt.show()
            f.savefig("./plots/errorbars_N_t.pdf", bbox_inches='tight')

        if R_percentile_store is None:
            fig = plt.figure(0)
            plt.plot(self.script_R, 1 - self.confidence_delta_R, label=r'$\mathcal{R}(\delta_\mathcal{R})$')
            plt.grid('True')
            plt.xlabel(r'Total Charging Rate of Active Users, $\mathcal{R}$ (kWh)')
            plt.ylabel(r"$\mathbb{P}\left(Q(t) < \mathcal{R}\right)$")
            plt.ylim([0, 1.05])
            # plt.title(r'Total Charging Rate vs. Confidence Interval')
            plt.show()
        else:
            spacing=5
            R_percentile_store = np.delete(R_percentile_store, [0], axis=0)
            R_percentile_means = np.mean(R_percentile_store, axis=0)

            f = plt.figure(0)
            plt.plot(self.script_R[2:], 1 - self.confidence_delta_R[2:], label=r'Theoretical Bound', color=(74.0/255.0, 176.0/255.0, 166.0/255))
            plt.plot(R_percentile_means, percentiles2/100, label=r'$Q(t)$ Monte Carlo',color=(93.0/255.0, 58.0/255.0, 155.0/255))

            R_percentile_store = R_percentile_store[:, np.arange(1,97, spacing)-1]
            percentiles2 = np.arange(1,97,spacing)
            R_percentile_means = np.mean(R_percentile_store, axis=0)
            R_percentile_std = np.std(R_percentile_store, axis=0)
            xerr = [2*R_percentile_std, 2*R_percentile_std]

            plt.errorbar( R_percentile_means, percentiles2/100, xerr=xerr, color=(93.0/255.0, 58.0/255.0, 155.0/255),fmt='none')
            plt.grid('True')
            plt.xlabel(r'Total Charging Rate of Active Users, $\mathcal{R}$ (kWh)')
            plt.ylabel(r"$\mathbb{P}\left(Q(t) < \mathcal{R}\right)$")
            plt.ylim([0, 1.05])
            plt.xlim([0.0, max(self.script_R)])
            # plt.title(r'Total Charging Rate vs. Confidence Interval')
            plt.legend(loc='lower right')
            plt.rcParams.update({'font.size': 14})    
            plt.show()
            f.savefig("./plots/errorbars_Q_t.pdf", bbox_inches='tight')

    def run_simulation(self):
        """
            This function runs the charging facility simulation.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         
        """
        
        self.UserList = []
        self.time_for_plot = np.linspace(0, self.T, self.num_t_samples)
        users_arrived = np.random.poisson(self.lambda_val*self.T)
        time_range = np.sort(self.T*np.random.rand(users_arrived))
        t = np.tile(self.time_for_plot, (users_arrived, 1))     # time range
        self.usermatrix = np.zeros([users_arrived, self.num_t_samples])     
        
        # Get Active Users
        for a_j in time_range:
            # Draw charging demand
            if self.x_distribution == 'uniform':
                user_xj = np.random.uniform(min(self.x_minmax), max(self.x_minmax))
            elif self.x_distribution == 'trunc_normal':
                lower = min(self.x_minmax)
                upper = max(self.x_minmax)
                mu = (upper + lower)/2
                sigma = (upper - mu)*.5
                user_xj = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)
                
            # Draw impatience factor
            if self.alpha_distribution == 'uniform':
                user_alphaj = np.random.uniform(min(self.alpha_minmax), max(self.alpha_minmax))
            elif self.alpha_distribution == 'trunc_normal':
                lower = min(self.alpha_minmax)
                upper = max(self.alpha_minmax)
                mu = (upper + lower)/2
                sigma = (upper - mu)*.5
                user_alphaj = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)
            
            if self.model != "specialcase":
                # Draw charging demand
                if self.xi_distribution == 'uniform':
                    user_xij = np.random.uniform(min(self.xi_minmax), max(self.xi_minmax))

            # Get chosen rate by user
            if self.model == "special":
                user_rj = self._get_price_func_param(user_xj, user_alphaj)
                self.UserList.append(User(a_j, user_xj, user_alphaj, user_rj))
            else:
                user_rj = self._get_price_func_param(user_xj, user_alphaj, user_xij)
                self.UserList.append(User(a_j, user_xj, user_alphaj, user_rj, user_xij))
                
        time_range = np.tile(time_range, (self.num_t_samples, 1)).transpose()
       
        charge_time = np.empty([])
        present_time = np.empty([])
        user_rate = np.empty([])
        
        # Loop through user list to get charge time & user rate 
        for user in self.UserList:
            charge_time = np.append(charge_time, user.u_j)
            present_time = np.append(present_time, user.c_j)
            user_rate = np.append(user_rate, user.r_j)

        # Erase first zero & make charge_time (user rate) a matrix
        charge_time = np.delete(charge_time, 0)
        present_time = np.delete(present_time, 0)
        user_rate = np.delete(user_rate, 0)

        charge_time =  np.tile(charge_time, (self.num_t_samples, 1)).transpose()
        present_time =  np.tile(present_time, (self.num_t_samples, 1)).transpose()
        user_rate = np.tile(user_rate, (self.num_t_samples, 1)).transpose()
        
        # Assign rows & column indices for times where each vechile is in the system
        rows = np.nonzero((t >= time_range) & (t <= time_range + present_time))[0]
        cols = np.nonzero((t >= time_range) & (t <= time_range + present_time))[1]
        
        # Assign rows & column indices for times where each vechile is actively charging in system
        rows_act = np.nonzero((t >= time_range) & (t <= time_range + charge_time))[0]
        cols_act = np.nonzero((t >= time_range) & (t <= time_range + charge_time))[1]

        # Define mask for present vehicles
        mask = np.ones(t.shape, dtype=bool)
        mask[rows, cols] = False
        
        # Define mask for actively charging vehicles
        mask_act = np.ones(t.shape, dtype=bool)
        mask_act[rows_act, cols_act] = False

        self.usermatrix[rows, cols] = 1
        user_rate[mask_act] = 0 # set times when user not active to zero
        
        self.activeusers =  np.sum(self.usermatrix, axis=0)
        self.totalrate = np.sum(user_rate, axis=0)



class ChargingFacilityContinuous:
    """
        Member functions of the charging facility class for a continuous pricing scheme. Contains functions & variables 
        for EV simulation.
    """
    
    def __init__(self, D, B, gamma, tau, R_max, T, alpha_list, x_list, xi_list, lambdaval,  num_t_samples, model):
        """
            This function initializes the user class.
        
        Parameters
        ----------
            D : numpy array
                y-intercept of pricing function
            B : list
                charging rate (i.e., inverse of pricing function slope)
            gamma: float
                simulation total time (hrs.)
            tau: float

            alpha_list: list
                contains min alpha, max alpha, and alpha's distribution (impatience factor, $/hr)
            x_list: list
                contains min x, max x, and x's distribution (charging demand, kWh)
            xi_list: list
                contains min xi, max xi, and xi's distribution (time at location, hr)
            lambaval: list
                value of lambda 
            num_t_samples: list
                number of time samples
            model: string
            	model used, changes pricing function in continuous case
                
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
             
        """
        self.model = model
        self.D = D
        self.B = B
        self.gamma = gamma
        self.tau = tau
        self.R_max = R_max
        self.num_t_samples = num_t_samples
        self.T = T

        # Pull min & max values of demand & impatience
        self.alpha_minmax = np.array(alpha_list[0:2])
        self.x_minmax = np.array(x_list[0:2])
        self.xi_minmax = np.array(xi_list[0:2])

        # Distributions STrings Stored
        self.alpha_distribution = alpha_list[2]
        self.x_distribution = x_list[2]
        self.xi_distribution = xi_list[2]

        self._check_surge_price()
        # # Compute Expectation
        # self.exp_uj = self._get_expectation_uj()
        # print(self.exp_uj)
        self.exp_uj, self.exp_rj,\
                self.exp_second_moment_rj = self.compute_expectations_numberical()
        self.lambda_val = lambdaval
        self.exp_eta = lambdaval*self.exp_uj
        # self._get_expectation_second_moment_rj()
        
        self.r_coeff = np.array([R_max]) 
        # # Compute Upper Bounds
        self._compute_M_delta()
        self._compute_R_delta()

    
    def _check_surge_price(self):
        """
            This function checks to see if the input surge price into the simulation adheres to the 
            surge price constraint such that the maximum charge rate constraint is respect.

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
            n/a
        """

        def denominatorfunction(x):
            return -(2*self.gamma * self.R_max * x - 2*x**2)

        result = optimize.minimize_scalar(denominatorfunction)
        denmax = denominatorfunction(result.x) * -1

        surge_lower_bound =  (min(self.alpha_minmax * self.tau**2 * self.R_max))/denmax

        # print("The surge price D must obey: D > ", surge_lower_bound)

        # if self.D > surge_lower_bound:
        #     print("Your entered surge price obeys the constraint")
        if self.D < surge_lower_bound:
            print("The surge price D must obey: D > ", surge_lower_bound)
            sys.exit("Please select another surge price. This one does not obey the constraint") 



    def _get_deadline_choice(self, user_xj, user_alphaj, user_xij):
        """
            This function computes the deadline choice for users.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        return np.maximum(user_xij, (-user_alphaj * self.tau**2)/(2 * self.D * user_xj)+ self.gamma )
    
    def _CDF_alphaj(self, alphaj):
        """
            This function computes the CDF of x_j
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """

        cdf_alphaj = quad(self._PDF_alphaj, 0, alphaj)[0]

        return cdf_alphaj

    def _PDF_alphaj(self, alphaj):
        """
            This function is the PDF of xj.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        if self.alpha_distribution == "uniform":
            alphamin = min(self.alpha_minmax)
            alphamax = max(self.alpha_minmax)

            pdf_alpha = 1/(alphamax - alphamin)  * (alphamin < alphaj) * (alphaj < alphamax)

        return pdf_alpha

    def _CDF_xj(self, xj):
        """
            This function computes the CDF of x_j
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """

        cdf_xj = quad(self._PDF_xj, 0, xj)[0]

        return cdf_xj

    def _PDF_xj(self, xj):
        """
            This function is the PDF of xj.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        if self.x_distribution == "uniform":
            xmin = min(self.x_minmax)
            xmax = max(self.x_minmax)

            pdf_X = 1/(xmax - xmin)  * (xmin < xj) * (xj < xmax)

        return pdf_X 
    
    def _CDF_xij(self, xij):
        """
            This function computes the PDF of x_j/r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """

        cdf_xij = quad(self._PDF_xij, 0, xij)[0]

        return cdf_xij

    def _PDF_xij(self, xij):
        """
            This function is the PDF of xi_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        if self.xi_distribution == "uniform":
            ximin = min(self.xi_minmax)
            ximax = max(self.xi_minmax)

            pdf_Xi = 1/(ximax - ximin) * (ximin < xij) * (xij < ximax)

        return pdf_Xi 

    def _CDF_uj(self, uj):
        """
            This function computes the cumulative distribution of uj.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """

        cdf_uj = quad(self._PDF_uj, 0, uj)[0]

        return cdf_uj

    def _PDF_uj(self, uj):
        """
            This is the probability density function of of uj.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        # pass
        # pdf_uj = 0.0
        pdf_uj = self._PDF_xij(uj)*self._CDF_uj_subcase(uj) + self._CDF_xij(uj) * self._PDF_uj_subcase(uj)

        return pdf_uj 

    def _CDF_uj_subcase(self, uj):
        """
            This function computes the cumulative distribution of the right subcase of uj.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """

        cdf_uj_sub = quad(self._PDF_uj_subcase, 0.01, uj)[0]

        return cdf_uj_sub

    def _PDF_uj_subcase(self, uj_sub):
        """
            This is the probability density function of right subcase of uj i.e., .
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        pdf_uj_subcase = (self.D * 2)/self.tau**2 * self._PDF_alphaj_xj( (-self.D * 2)/self.tau**2 * (uj_sub - self.gamma))

        return pdf_uj_subcase

    def _CDF_alphaj_xj(self, alphaj_xj):
        """
            This is the cumulative distribution function of alpha_j/x_j.
        
        Parameters
        ----------
            alphaj_xj: n

            
        Returns
        -------
            cdj_alpha_xj: float
                cumulative distribution value at alphaj_xj
        
        Notes
        -----
            n/a
         """

        cdf_alphaj_xj = quad(self._PDF_alphaj_xj, 0, alphaj_xj)[0]

        return cdf_alphaj_xj

    def _PDF_alphaj_xj(self, alphaj_xj):
        """
            This is the probability density function of alpha_j/x_j
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        x_min = min(self.x_minmax)
        x_max = max(self.x_minmax)
        alpha_min = min(self.alpha_minmax)
        alpha_max = max(self.alpha_minmax)

        return (np.minimum(x_max, alpha_max/alphaj_xj)**2 - np.maximum(x_min, alpha_min/alphaj_xj)**2)\
                /(2 * (x_max - x_min) * (alpha_max - alpha_min)) 


    def _get_expectation_cj(self):
        """
            This function computes the expectation of the random variable cj = max(xi_j, xj/rj).
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        ## TODO: Write code for expectation
        self.cj = self.uj 

        return self.cj

    def _get_expectation_xj(self):
        """
            This function computes the expectation .
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
         
        def X_first_moment(X):
            if self.x_distribution == "uniform":
                val = X*self._PDF_xj(X)
            
            return val
        
        exp_xj = quad(X_first_moment, min(self.x_minmax), max(self.x_minmax))[0]

        self.exp_xj = exp_xj

        return self.exp_xj

    def _get_expectation_rj(self):
        """
            This computes the expect value of the charging rate, r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """

        ### TODO: Add code for expectation continuous
        self.exp_rj = 0

        
        return self.exp_rj

    def _get_expectation_uj(self):
        """
            Computes the expected value of the time to charge.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            exp_xj/exp_j: numpy array
                
                expectation of u, i.e., expectation of time to charge
        
        Notes
        -----
            n/a
         """

        def U_first_moment(U):
            val = U*self._PDF_uj(U)
            
            return val
        
        exp_uj = quad(U_first_moment, 0.01, 15)[0]

        self.exp_uj = exp_uj

        return self.exp_uj

        exp_xj = self._get_expectation_xj()
        exp_rj = self._get_expectation_rj()
         
        return exp_xj/exp_rj
    
    
    def _get_expectation_second_moment_rj(self):
        """
            This computes the second moment of the charging rate, r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        r_index = 0
        self.exp_second_moment_rj = 0
        for probability in self.price_func_min_probability:
            self.exp_second_moment_rj = self.exp_second_moment_rj + probability*self.r_coeff[r_index]**2
            r_index = r_index + 1

    def compute_expectations_numberical(self):
        """
            This function computes a sanity check on the computed values.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        ## TODO: WRITE SANITY CHECK CODE
        elements = 5000
        X_array = np.random.uniform(min(self.x_minmax), max(self.x_minmax), elements)
        Alpha_array = np.random.uniform(min(self.alpha_minmax), max(self.alpha_minmax), elements)
        Xi_array = np.random.uniform(min(self.xi_minmax), max(self.xi_minmax), elements)

        U_array = self._get_deadline_choice(X_array, Alpha_array, Xi_array)
        R_array = np.divide(X_array, U_array)

        self.exp_uj = np.mean(U_array)
        self.exp_rj = np.mean(R_array)
        self.exp_second_moment_rj = np.mean(R_array**2)
        # print("Numerical Sanity Check")
        # print("E[u]:", np.mean(U_array))
        # print("E[r]:", np.mean(R_array))
        return self.exp_uj, self.exp_rj, self.exp_second_moment_rj

    def compute_sanity_check(self):
        """
            This function computes a sanity check on the computed values.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        ## TODO: WRITE SANITY CHECK CODE
        elements = 5000
        X_array = np.random.uniform(min(self.x_minmax), max(self.x_minmax), elements)
        Alpha_array = np.random.uniform(min(self.alpha_minmax), max(self.alpha_minmax), elements)
        Xi_array = np.random.uniform(min(self.xi_minmax), max(self.xi_minmax), elements)

        U_array = self._get_deadline_choice(X_array, Alpha_array, Xi_array)
        R_array = np.divide(X_array, U_array)

        print("Numerical Sanity Check")
        print("E[u]:", np.mean(U_array))
        print("E[r]:", np.mean(R_array))
        print("E[r^2]", np.mean(R_array**2))

    def _compute_M_delta(self):
        """
            This function computes the probability of a function being the min
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
#        self.delta_M_range = np.linspace(1e-2, 1, 100)
#        self.M_func = self.exp_eta + 1/3 * np.log(np.divide(1, self.delta_M_range)) + \
#        np.sqrt(1/9 * np.log(np.divide(1, self.delta_M_range))**2 + \
#        2*self.exp_eta*np.log(np.divide(1, self.delta_M_range)))
#        
#        self.M_func_loose = self.exp_eta + 2/3 * np.log(np.divide(1, self.delta_M_range)) + \
#        np.sqrt(2*self.exp_eta*np.log(np.divide(1, self.delta_M_range)))
        self.confidence_delta_M = np.empty([])
        self.script_M = np.linspace(.05, 100, 1000) + self.exp_eta

#        temp_delta_M = 0.0
#        for M_val in self.script_M:s
        numerator = -(self.script_M - self.exp_eta)**2
        denominator = 2*(self.exp_eta + 1/3 * (self.script_M - self.exp_eta))
        
        self.confidence_delta_M  = np.exp(np.divide(numerator, denominator))

    def _compute_R_delta(self):
        """
            This function computes the probability of a function being the min
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """ 
        self.script_R = np.linspace(self.exp_eta*self.exp_rj,2500,20)
  
        self.confidence_delta_R = np.empty([])

        for script_R_values in self.script_R:
            sum_temp_var = 0.0 # temporary variable to perform summation
            
            # Define lower and upper bound of summation
            M_upper = int(np.floor(script_R_values/self.exp_rj))
            M_lower = int(np.ceil(script_R_values/max(self.r_coeff)))
#            M_lower = 0
            
            # Define numerator of and denominator of exponential
            numerator = -(M_upper - self.exp_eta)**2     
            denominator = 2*(self.exp_eta + 1/3 * (M_upper - self.exp_eta))
            
            # Define m values over which to perform sum
            m = np.linspace(M_lower, M_upper, M_upper - M_lower + 1)
            
            exp_numerator = -(script_R_values - m*self.exp_rj)**2
            exp_denominator = 2*(m*self.exp_second_moment_rj + (1/3)*max(self.r_coeff)*(script_R_values - m*self.exp_rj))
            sum_total_probability = np.multiply(np.exp(np.divide(exp_numerator, exp_denominator)), \
                                    stats.poisson.pmf(m, self.exp_eta, loc = 0))
            
            if script_R_values <= self.exp_eta*self.exp_rj:
                sum_temp_var = 1
            else:
                sum_temp_var = np.sum(sum_total_probability) +  np.exp(np.divide(numerator, denominator))
                sum_temp_var= min(1, sum_temp_var)


#            print(sum_temp_var)
            self.confidence_delta_R = np.append(self.confidence_delta_R, sum_temp_var)
#        self.script_R = self.script_R + self.exp_eta*self.exp_rj
        self.confidence_delta_R = np.delete(self.confidence_delta_R, [0], axis=0)
        
    def get_pricingfunc_plot(self, DistFlag=None):
        """
            This function prints out all the pricing functions and saves them.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         
        """
        
        if self.model == "quadratic":
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            # Make data.
            U = np.arange(0, 10, 0.25)
            Xi = np.arange(min(self.xi_minmax),max(self.xi_minmax), 0.25)
            U_mesh, Xi_mesh = np.meshgrid(U, Xi)
            TotalCost = self.D * (U_mesh - self.gamma)**2/self.tau**2 + (U_mesh - Xi_mesh)

            # Plot the surface.
            surf = ax.plot_surface(U_mesh, Xi_mesh, TotalCost, cmap=cm.coolwarm,linewidth=0, antialiased=False, alpha=.5)
            SIZE = 15
            ax.set_title(r'Illustration of Total Cost Function')
            ax.set_xlabel(r'$u_j$', labelpad=10, size=SIZE)
            ax.set_ylabel(r'${\xi}_j$',labelpad=10, size=SIZE)
            ax.set_zlabel(r'$C\left(x_j, \cdot, \alpha_j, \cdot \right)$',labelpad=10, size=SIZE)

            # Add a color bar which maps values to colors.
            cbaxes = fig.add_axes([1.0, 0.22, 0.03, 0.5]) 

            fig.colorbar(surf, shrink=0.5, aspect=5, cax = cbaxes, alpha=.5)
            plt.gcf().subplots_adjust(bottom=0.25)
            plt.show()
            fig.savefig("./plots/pricingfunc_continuous.pdf", bbox_inches='tight')


    def get_sim_plots(self):
        """
            This function runs the charging facility simulation.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
         
        
        f = plt.figure()
        plt.plot(self.time_for_plot,self.activeusers)
        plt.grid('True')
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\eta(t)$")
        plt.title(r"Number of users versus Time (hr.)")
        plt.show()
        f.savefig("./plots/num_of_users_vs_t.pdf", bbox_inches='tight')
        
        f = plt.figure()
        plt.plot(self.time_for_plot, self.totalrate)
        plt.grid('True')
        plt.xlabel(r"$t$")
        plt.ylabel(r"$R(t)$")
        plt.title(r"Charging Rate (kW) versus Time (hr.)")
        plt.show()
        f.savefig("./plots/chargerate_vs_t.pdf", bbox_inches='tight')
        
    def get_upperbound_plot(self, N_percentile_store=None, percentiles=None, R_percentile_store=None, percentiles2=None):
        """
            Plots the results from the 
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        if N_percentile_store is None:
            fig = plt.figure(0)
            plt.plot(1 - self.delta_M_range, self.M_func)
            plt.plot(1 - self.delta_M_range, self.M_func_loose)
            plt.grid('True')
            plt.xlabel(r'Number of Present Users, $\mathcal{M}$')
            plt.ylabel(r"$\mathbb{P}\left(\eta(t) < \mathcal{M}\right)$")
            plt.ylim([0, 1.05])
            plt.show()
        else:
            spacing=5
            N_percentile_store = np.delete(N_percentile_store, [0], axis=0)
            N_percentile_means = np.mean(N_percentile_store, axis=0)

            f = plt.figure(0)
            plt.plot(self.script_M, 1- self.confidence_delta_M, label=r'Theoretical Bound', color=(0.0/255.0, 90.0/255.0, 181.0/255))
            plt.plot(N_percentile_means, percentiles/100, label=r'$\eta(t)$ Monte Carlo', color=(220.0/255.0, 50.0/255.0, 32.0/255))
            
            N_percentile_store = N_percentile_store[:, np.arange(1,97, spacing)-1]
            percentiles = np.arange(1,97,spacing)
            N_percentile_means = np.mean(N_percentile_store, axis=0)
            N_percentile_std = np.std(N_percentile_store, axis=0)
            # xerr = [N_percentile_store.min(axis=0)-N_percentile_means, N_percentile_means-N_percentile_store.max(axis=0)]
            xerr = [2*N_percentile_std, 2*N_percentile_std]

            plt.errorbar(N_percentile_means, percentiles/100, xerr=xerr, color=(220.0/255.0, 50.0/255.0, 32.0/255), fmt='none')
            plt.grid('True')

            plt.xlabel(r'Number of Present Users, $\mathcal{M}$')
            plt.ylabel(r"$\mathbb{P}\left(\eta(t) < \mathcal{M}\right)$")

            plt.xlim([0, max(self.script_M)])
            plt.ylim([0, 1.05])
            # plt.title(r'Total Active Users vs. Confidence Interval')
            plt.legend(loc=4)
            plt.rcParams.update({'font.size': 14})    
            plt.show()
            f.savefig("./plots/errorbars_N_t_continuous.pdf", bbox_inches='tight')

        if R_percentile_store is None:
            fig = plt.figure(0)
            plt.plot(self.script_R, 1 - self.confidence_delta_R, label=r'$\mathcal{R}(\delta_\mathcal{R})$')
            plt.grid('True')
            plt.xlabel(r'Total Charging Rate of Active Users, $\mathcal{R}$ (kWh)')
            plt.ylabel(r'$\mathbb{P}\left(Q(t) < \mathcal{R}\right)$')
            plt.ylim([0, 1.05])
            plt.title(r'Total Charging Rate vs. Confidence Interval')
            plt.show()
        else:
            spacing=5
            R_percentile_store = np.delete(R_percentile_store, [0], axis=0)
            R_percentile_means = np.mean(R_percentile_store, axis=0)
            f = plt.figure(0)
            plt.plot(self.script_R[1:], 1 - self.confidence_delta_R[1:], label=r'Theoretical Bound', color=(0.0/255.0, 90.0/255.0, 181.0/255))
            plt.plot(R_percentile_means, percentiles2/100, label=r'$Q(t)$ Monte Carlo', color=(220.0/255.0, 50.0/255.0, 32.0/255))

            R_percentile_store = R_percentile_store[:, np.arange(1,97, spacing)-1]
            percentiles2 = np.arange(1,97,spacing)
            R_percentile_means = np.mean(R_percentile_store, axis=0)
            R_percentile_std = np.std(R_percentile_store, axis=0)
            # xerr = [R_percentile_store.min(axis=0)-R_percentile_means, R_percentile_means-R_percentile_store.max(axis=0)]
            xerr = [2*R_percentile_std, 2*R_percentile_std]

            plt.errorbar( R_percentile_means, percentiles2/100, xerr=xerr, color=(220.0/255.0, 50.0/255.0, 32.0/255), fmt='none')
            plt.grid('True')
            plt.xlabel(r'Total Charging Rate of Active Users, $\mathcal{R}$ (kWh)')
            plt.ylabel(r'$\mathbb{P}\left(Q(t) < \mathcal{R}\right)$')
            plt.ylim([0, 1.05])
            plt.xlim([0.0, max(self.script_R)])
            # plt.title(r'Total Charging Rate vs. Confidence Interval')
            plt.legend(loc=4)
            plt.rcParams.update({'font.size': 14})    
            plt.show()
            f.savefig("./plots/errorbars_Q_t_continuous.pdf", bbox_inches='tight')

    def run_simulation(self):
        """
            This function runs the charging facility simulation.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         
        """
        
        self.UserList = []
        self.time_for_plot = np.linspace(0, self.T, self.num_t_samples)
        users_arrived = np.random.poisson(self.lambda_val*self.T)
        time_range = np.sort(self.T*np.random.rand(users_arrived))
        t = np.tile(self.time_for_plot, (users_arrived, 1))     # time range
        self.usermatrix = np.zeros([users_arrived, self.num_t_samples])     
        
        # Get Active Users
        for a_j in time_range:
            # Draw charging demand
            if self.x_distribution == 'uniform':
                user_xj = np.random.uniform(min(self.x_minmax), max(self.x_minmax))
            # elif self.x_distribution == 'trunc_normal':
            #     lower = min(self.x_minmax)
            #     upper = max(self.x_minmax)
            #     mu = (upper + lower)/2
            #     sigma = (upper - mu)*.5
            #     user_xj = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)
                
            # Draw impatience factor
            if self.alpha_distribution == 'uniform':
                user_alphaj = np.random.uniform(min(self.alpha_minmax), max(self.alpha_minmax))
            # elif self.alpha_distribution == 'trunc_normal':
            #     lower = min(self.alpha_minmax)
            #     upper = max(self.alpha_minmax)
            #     mu = (upper + lower)/2
            #     sigma = (upper - mu)*.5
            #     user_alphaj = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)
                
            # Draw time spend at charging location
            if self.xi_distribution == 'uniform':
                user_xij = np.random.uniform(min(self.xi_minmax), max(self.xi_minmax))
            # elif self.xi_distribution == 'trunc_normal':
            #     lower = min(self.xi_minmax)
            #     upper = max(self.xi_minmax)
            #     mu = (upper + lower)/2
            #     sigma = (upper - mu)*.5
            #     user_xij = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)

            user_uj = self._get_deadline_choice(user_xj, user_alphaj, user_xij)
            self.UserList.append(UserCont(a_j, user_xj, user_alphaj, user_xij, user_uj))
                
        time_range = np.tile(time_range, (self.num_t_samples, 1)).transpose()
       
        charge_time = np.empty([])
        user_rate = np.empty([])
        
        # Loop through user list to get charge time & user rate 
        for user in self.UserList:
            charge_time = np.append(charge_time, user.u_j)
            user_rate = np.append(user_rate, user.r_j)
            
        # Erase first zero & make charge_time (user rate) a matrix
        charge_time = np.delete(charge_time, 0)
        user_rate = np.delete(user_rate, 0)
        charge_time =  np.tile(charge_time, (self.num_t_samples, 1)).transpose()
        user_rate = np.tile(user_rate, (self.num_t_samples, 1)).transpose()
        
        # Assign rows & column indices for times where each vechile is in the system
        rows = np.nonzero((t >= time_range) & (t <= time_range + charge_time))[0]
        cols = np.nonzero((t >= time_range) & (t <= time_range + charge_time))[1]
        
        # Define mask
        mask = np.ones(t.shape, dtype=bool)
        mask[rows, cols] = False
        
        self.usermatrix[rows, cols] = 1
        user_rate[mask] = 0 # set times when user not active to zero
        
        self.activeusers =  np.sum(self.usermatrix, axis=0)
        self.totalrate = np.sum(user_rate, axis=0)


