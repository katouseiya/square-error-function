import computeCost_2 as com
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def gradientDescent(X,y,theta,alpha,num_iters):
    m = y.shape[0]
    theta_X = theta.shape[0]
    J = np.zeros((num_iters,1,))
    kazu = np.zeros((num_iters,1,))
    
    for iter in range(num_iters):
        delta = np.zeros((theta_X,1,))
               
        for j in range(m):
            h = 0
            for k in range(theta_X):
                h += theta[k,0] * X[j,k]
            for u in range(theta_X):
                delta[u,0] += float(((h - y[j,0])*X[j,u])*(1/m))
        for n in range(theta_X):
            theta[n,0] = theta[n,0] - (alpha)*(delta[u,0])

        J[iter,0] = com.costFunctionJ(X,y,theta)
        kazu[iter,0] = iter
    print(J)
    plt.plot(kazu,J)
    plt.show()
    
        
        
                

        
        
