# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:54:17 2020

@author: Lando
"""

import math
import os
import pandas as pd
import numpy as np



os.chdir("C:\\Users\\Lando\\Desktop\\GitHub\\Newton-Raphson-and-GLRT")



data = pd.read_csv("melanoma years till death or relapse.csv")
X = data.X

### Finding MLES through Newton Rhapson algorithm
# =========================================================================
def nwt(func,Dfunc,x0,min,max_iter):
    '''
    
    Parameters
    ----------
    func : 
        This is the function that needs a root found. If trying to estimate a MLE, this is
        the derivative of the log likelihood as we want to maximize the log likelihood.
    Dfunc : function
        The derivative of the function func.
    x0 : float
        This is the starting point for the algorithm. Should be a best guess of the root.
    min : float
        Threshold for stopping criteria
    max_iter : int
        maximum number of rounds before stopping.

    Returns
    -------
    xn : 
        Root of func

    '''
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
# =========================================================================


### Building Confidence Intervals
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






