# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:18:46 2020

@author: Lando
"""


import math
import os
import pandas as pd
import numpy as np
from scipy import stats # For chi-square

os.chdir("C:\\Users\\Lando\\Desktop\\Newton Rhapson")
data = pd.read_csv("melanoma years till death or relapse.csv")
X = data.X

### Finding MLES through Newton Rhapson
# =========================================================================
def nwt(func,Dfunc,x0,min,max_iter):
    xn = x0
    for n in range(0,max_iter):
        fx = func(xn)
        Dfx = Dfunc(xn)
        if Dfx == 0:
            print('Divide by Zero Error')
            return None
        xm = xn - fx/Dfx
        diff = xm - xn
        if abs(diff) < min:
            print(f'Found solution after {n} iterations.')
            return xn
        else:
            xn = xm
    print('Exceeded maximum iterations. No solution found.')
    return None


fw = lambda B: math.fsum(X**B*np.log(X))/math.fsum(X**B) - 1/B - math.fsum(np.log(X))/len(X)
dfw = lambda B: math.fsum(np.log(X)**2*X**B)/math.fsum(X**B) + 1/B**2 + 1
B = nwt(fw,dfw,2 ,1e-8, 100)
A = math.fsum(X**B/len(X))**(1/B)
print(f'Our estimate of Beta is: {round(B, 4)}')
print(f'Our estimate of Theta is: {round(A, 4)}')


# SE should be .01576,
# 0.1078 = SE, 1.96*SE   -- according to allen

# =========================================================================


### Building Confidence Intervals for B & A, then plug in for Median
# =========================================================================

## Observed Information Matrix
n = len(X)
## Get all Seconde Derivatives used in information matrix
dbb = -n/B**2 - math.fsum(((X/A)**B)*np.log(X/A)**2)
dba = -n/A + (B/A)*math.fsum(((X/A)**B)*np.log(X/A)) + (1/A)*math.fsum((X/A)**B)

dab = -n/A + (1/A**(B+1))*math.fsum(X**B) + (1/A**(B+1))*(math.fsum((X**B)*B*np.log(X/A)))
daa = n*B/(A**2) + (-1-B)*B*A**(-2-B)*math.fsum(X**B)


## Get observed information matrix
m = -1*np.matrix([[dbb,dba],[dab,daa]])

## get Variance Covariance Matrix by inverting information matrix
vc = m.I

## the top left, bottom right are the variances, the other 2 are covariances. Use the variances for conf. intervals.
B_lower = B - 1.96*np.sqrt(vc[0,0])
B_upper = B + 1.96*np.sqrt(vc[0,0])

A_lower = A - 1.96*np.sqrt(vc[1,1])
A_upper = A + 1.96*np.sqrt(vc[1,1])


B_list = [B_lower, B, B_upper]
A_list = [A_lower, A, A_upper]

## Print out MLEs with Confidence Intervals.
print(f"Beta MLE [lower CI, estimate, upper CI]:\n  {B_list}")
print(f"Theta MLE [lower CI, estimate, upper CI]:\n  {A_list}")


#### Median Derivation


## Finding the Median
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

v_ho = weibull_lik(1, To, X) # Theta is set to the MLE Theta in both. Alternative would be to recalcualte A based on B being 1.. Not sure which is right.
v_mle = weibull_lik(B, A, X)
v = v_ho/v_mle
ch_stat = -2*np.log(v)
pv = 1 - stats.chi2.cdf(ch_stat, 1)

print(f'''
Test Statistic: {ch_stat}
P-Value: {pv}   
      ''')


