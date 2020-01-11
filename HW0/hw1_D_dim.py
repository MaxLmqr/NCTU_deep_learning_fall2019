import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io

# Read files
mat = scipy.io.loadmat('Iris_X.mat')
matY = scipy.io.loadmat('Iris_T.mat')
X = mat['X']
Y = matY['T']

# Build training set and test set
Xtrain = X[:40,:].tolist() + X[50:90,:].tolist() + X[100:140,:].tolist()
Xtest = X[40:50,:].tolist() + X[90:100,:].tolist() + X[140:150,:].tolist()
Ytrain = Y[:40,:].tolist() + Y[50:90,:].tolist() + Y[100:140,:].tolist()
Ytest = Y[40:50,:].tolist() + Y[90:100,:].tolist() + Y[140:150,:].tolist()

# Construct matrix A 
A = Xtrain.copy()
A = [[1] + x for x in A]
for i in range(len(Xtrain[0])):
    for j in range(len(Xtrain[0])):
        A = [x + [x[i]*x[j]] for x in A]
for i in range(len(Xtrain[0])):
    for j in range(len(Xtrain[0])):
        for k in range(len(Xtrain[0])):
           A = [x + [x[i]*x[j]*x[k]] for x in A]
# Matrix A looks like [ x1, ..., xD, x1x1, x1x2, ...., xDxD, x1x1x1, ....]
A_array = np.asarray(A)

# Construct matrix A test
Atest = Xtest.copy()
Atest = [[1] + x for x in Atest]
for i in range(len(Xtest[0])):
    for j in range(len(Xtest[0])):
        Atest = [x + [x[i]*x[j]] for x in Atest]
for i in range(len(Xtest[0])):
    for j in range(len(Xtest[0])):
        for k in range(len(Xtest[0])):
           Atest = [x + [x[i]*x[j]*x[k]] for x in Atest]
# Matrix Atest looks like [ x1, ..., xD, x1x1, x1x2, ...., xDxD, x1x1x1, ....]
A_test_array = np.asarray(Atest)


def compute_testError(Xtest, Ytest, Atest, curve_coefficient):

    testError = np.dot(np.transpose(np.dot(Atest, curve_coefficient) - Ytest), (np.dot(Atest, curve_coefficient) - Ytest))
    return testError
#####


def lin_reg(Xtrain, Ytrain, A, lbda = 0):
    n = np.shape(A)[1]
    # Compute A transpose
    At = np.transpose(A)
    #####

    # Compute At * A + lbda*I and decompose it into L*U
    U = np.dot(At, A) + lbda/2*np.eye(n)

    # I couldn't compute myself the inverse because the matrix is singular
    # I found it out after quite a time, so
    # I used linalg to compute the Pseudo Inverse
    curve_coefficient = np.dot(np.linalg.pinv(U),np.dot(At,Ytrain))
     # Compute error
    error = np.dot(np.transpose(np.dot(A, curve_coefficient) -
                                Ytrain), (np.dot(A, curve_coefficient) - Ytrain))
    
    return curve_coefficient, error
#####


curve_coefficient, error = lin_reg(Xtrain, Ytrain, A_array)
testError = compute_testError(Xtest, Ytest, Atest, curve_coefficient)
print("Error on training set : ",error)
print("Error on testing set : ", testError)

