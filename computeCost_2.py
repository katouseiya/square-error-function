import numpy as np
def  costFunctionJ(X,y,theta):
    m = y.shape[0]
    predictions = np.zeros((m,1,))
    sqrErrors = np.zeros((m,1,))
    for i in range(m):
        predictions[i,0] = X[i,0]*theta[0,0] + X[i,1]*theta[1,0] + X[i,2]*theta[2,0]
        sqrErrors[i,0] = np.power((predictions[i,0] - y[i,0]),2)
    

    return sum(sqrErrors)/int(2*m)
