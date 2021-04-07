import numpy as np
def  costFunctionJ(X,y,theta):
    m = y.shape[0]
    predictions = np.zeros((m,1,))
    sqrErrors = np.zeros((m,1,))
    X_theta = theta.shape[0]
    for i in range(m):
        for j in range(X_theta):
            predictions[i,0] += X[i,j] * theta[j,0]
        sqrErrors[i,0] = np.power((predictions[i,0] - y[i,0]),2)
    

    return sum(sqrErrors)/int(2*m)
