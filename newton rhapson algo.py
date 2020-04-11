# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:54:17 2020

@author: Lando
"""

import math
import os
import pandas as pd
import numpy as np


os.chdir("C:\\Users\\Lando\\Desktop\\Newton Rhapson")
data = pd.read_csv("melanoma years till death or relapse.csv")


### Finding MLES through Newton Rhapson
# =========================================================================
def nwt(func,Dfunc,x0,min,max_iter):
    xn = x0
    for n in range(0,max_iter):
        fx = func(xn)
        if abs(fx) < min:
            print('Found solution after',n,'iterations.')
            return xn
        Dfx = Dfunc(xn)
        if Dfx == 0:
            print('Divide by Zero Error')
            return None
        xn = xn - fx/Dfx
    print('Exceeded maximum iterations. No solution found.')
    return None


X = data.X
fw = lambda B: math.fsum(X**B*np.log(X))/math.fsum(X**B) - 1/B - math.fsum(np.log(X))/len(X)
dfw = lambda B: math.fsum(np.log(X)**2*X**B)/math.fsum(X**B) + 1/B**2 + 1


B = nwt(fw,dfw,2 ,1e-8, 100)
A = math.fsum(X**B/len(X))**(1/B)
B
A
# =========================================================================

### Using some outside method to make sure it's working.
# ========================================================================

# Method 1
from scipy.stats import exponweib
from scipy.optimize import fmin

# x is your data array
# returns [shape, scale]
# Method 1
def fitweibull(x):
   def optfun(theta):
      return -np.sum(np.log(exponweib.pdf(x, 1, theta[0], scale = theta[1], loc = 0)))
   logx = np.log(x)
   shape = 1.2 / np.std(logx)
   scale = np.exp(np.mean(logx) + (0.572 / shape))
   return fmin(optfun, [shape, scale], xtol = 0.01, ftol = 0.01, disp = 0)

fitweibull(X)


# Method 2
from scipy import stats
check = stats.exponweib.fit(X, floc=0, f0=1)
check

# ========================================================================



### Building Confidence Intervals
# =========================================================================

## Observed Information Matrix
n = len(X)
## Get all Seconde Derivatives
dbb = -n/B**2 - math.fsum(((X/A)**B)*np.log(X/A)**2)
dba = -n/A + (B/A)*math.fsum(((X/A)**B)*np.log(X/A)) + (1/A)*math.fsum((X/A)**B)

dab = -n/A + (1/A**(B+1))*math.fsum(X**B) + (1/A**(B+1))*(math.fsum((X**B)*B*np.log(X/A)))
daa = n*B/(A**2) + (-1-B)*B*A**(-2-B)*math.fsum(X**B)


## Get observed information matrix
m = -1*np.matrix([[dbb,dba],[dab,daa]])

## get Variance Covariance Matrix by inverting m
vc = m.I

## the top left, bottom right are the variances, the other 2 are covariances. Use the variances for conf. intervals.
B_lower = B - 1.96*np.sqrt(vc[0,0])
B_upper = B + 1.96*np.sqrt(vc[0,0])

A_lower = A - 1.96*np.sqrt(vc[1,1])
A_upper = A + 1.96*np.sqrt(vc[1,1])


B_list = [B_lower, B, B_upper]
A_list = [A_lower, A, A_upper]


B_list
A_list






