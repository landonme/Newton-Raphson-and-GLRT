# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:18:46 2020

@author: Lando
"""

'''
This file calculates the theoretical median with 95% confidence interval
and runs a GLRT test under the null hypothesis that Beta = 1 for a Weibull Distribution.
'''


import math
import os
import pandas as pd
import numpy as np
from scipy import stats # For chi-square

os.chdir("C:\\Users\\Lando\\Desktop\\GitHub\\Newton-Raphson-and-GLRT")

import newton_rhapson_weibull_estimation


## Finding the theretical Median
## Note, this is not the actual median of the observed data, it's the theoretical
## Median based on the the MLEs.
def mfunc(b, a):
    return a*(np.log(2))**(1/b)
M = mfunc(B,A)
print(f'Our estimate of the Median is: {round(M, 4)}')

## 95% CI For Median
varb = vc[0,0]
vara = vc[1,1]
cov = vc[0,1]
dtheta = np.log(2)**(1/B)
dbeta = -(np.log(2)**(1/B)*np.log(np.log(2))*A)/(B**2)
dthetasq = dtheta**2
dbetasq = dbeta**2

var_m = dthetasq*vara + dbetasq*varb + 2*dtheta*dbeta*cov
se_m = np.sqrt(var_m)
M_upper = M + 1.96*se_m
M_lower = M - 1.96*se_m
M_list = [M_lower, M, M_upper]
print(f"Beta MLE [lower CI, estimate, upper CI]:\n  {M_list}")


### GLRT
# =========================================================================
# Test the hypothesis that β for these data is equal to 1, i.e.,
# that the data follow an exponential distribution, using the GLRT and asymptotic chi-squared approximation.
# Step 1: Find MLE under Ho and Ha.
# Under Ho, B = 1


# Function for Likelihood of Weibull Distribution
def weibull_lik(B, T, x):
    '''
    Parameters
    ----------
    B : Beta
    T : Theta
    x : Series of Data
    
    Returns the likelihood of the data given Beat & Theta
    -------
    '''
    n = len(x)
    return ((B/(T)**B)**n)*np.prod(x**(B-1))*np.exp(-math.fsum((x/T)**B))


To = math.fsum(X**1/len(X))**(1/1)

v_ho = weibull_lik(1, To, X)
v_mle = weibull_lik(B, A, X)
v = v_ho/v_mle
ch_stat = -2*np.log(v)
pv = 1 - stats.chi2.cdf(ch_stat, 1)

print(f'''
Test Statistic: {ch_stat}
P-Value: {pv}   
      ''')


