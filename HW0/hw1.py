import csv
import matplotlib.pyplot as plt
import numpy as np
import random

# Input parameters
n = 3       # Polynome degree
#####

# LU decomposition function
def LUdecomposition(A):
    U = A
    n = np.shape(A)[1]
    L = np.eye((n))
    for i in range(n):
        if U[i][i] == 0:                # If diagonal coefficient is equal to 0
            for k in range(i+1, n):  # check the lines below
                if U[k][i] != 0:
                    U[[k, i]] = U[[i, k]]     # Swap lines
        # Diagonal coefficient is now != 0
        for j in range(i):
            L[i][j] = U[i][j]/U[j][j]
            U[i] = U[i] - (U[i][j]/U[j][j])*U[j]  # cancel coefficient U[i][j]
    return L, U
#####

def lin_reg(Xtrain, Ytrain, n, lbda = 0):
    # Fill matrix A with data
    A = np.zeros((len(Xtrain), n+1))
    for i in range(n+1):
        for j in range(len(Xtrain)):
            A[j][i] = Xtrain[j]**i
    #####

    # Compute A transpose
    At = np.transpose(A)
    #####

    # Compute At * A + lbda*I and decompose it into L*U
    U = np.dot(At, A) + lbda/2*np.eye(n+1)
    L, U = LUdecomposition(U)
    #####

    # Solve system to compute curve_coefficient
    b = np.dot(At, Ytrain)
    # first system
    xtemp = np.zeros(len(L))
    for i in range(len(xtemp)):
        xtemp[i] = b[i]
        for j in range(i):
            xtemp[i] = xtemp[i] - L[i][j]*xtemp[j]
        xtemp[i] = xtemp[i]/L[i][i]

    # second system
    curve_coefficient = np.zeros(n+1)
    for i in range(len(curve_coefficient)):
        curve_coefficient[n-i] = xtemp[n-i]
        for j in range(i):
            curve_coefficient[n-i] = curve_coefficient[n-i] - \
                U[n-i][n-j]*curve_coefficient[n-j]
        curve_coefficient[n-i] = curve_coefficient[n-i]/U[n-i][n-i]

     # Compute error
    error = np.dot(np.transpose(np.dot(A, curve_coefficient) -
                                Ytrain), (np.dot(A, curve_coefficient) - Ytrain))

    return curve_coefficient, error
#####

def compute_testError(Xtest, Ytest, n, curve_coefficient):
    # Fill matrix Atest with data
    Atest = np.zeros((len(Xtest), n+1))
    for i in range(n+1):
        for j in range(len(Xtest)):
            Atest[j][i] = Xtest[j]**i
    #####

    testError = np.dot(np.transpose(np.dot(Atest, curve_coefficient) - Ytest), (np.dot(Atest, curve_coefficient) - Ytest))
    return testError
#####

# Read the data file
csvFile = open('data_1.csv', 'r')
reader = csv.reader(csvFile)
X = []
Y = []
for row in reader:
    try:
        X.append(float(row[0]))
        Y.append(float(row[1]))
    except:
        continue
#####

# Build the training set by picking 70% random value from the entire dataset
nTrainSet = round(0.7*len(X))
Xtrain = []
Ytrain = []
trainIndex = random.sample(range(0, len(X)), nTrainSet)
for w in trainIndex:
    Xtrain.append(X[w])
    Ytrain.append(Y[w])
#####

# Build the test set with the other values
Xtest = []
Ytest = []
for i in range(len(X)):
    if i not in trainIndex:  # If this index has not been used for the training set
        Xtest.append(X[i])
        Ytest.append(Y[i])
#####

# Compute curve_coefficient and error on the training set, also print the training error
curve_coefficient, error = lin_reg(Xtrain, Ytrain, n)
print("\nTotal Training Error : ", error)
#####

# Compute and print error on the testing set
testError = compute_testError(Xtest, Ytest, n, curve_coefficient)
print("\nTotal Test Error : ", testError)
#####

# Compute data to plot the linear regression
Abs = np.linspace(min(X), max(X), 1000)
MatAbs = [Abs**i for i in range(n+1)]
MatAbs = np.transpose(MatAbs)
Yreg = np.dot(MatAbs, curve_coefficient)
#####

# rootMeanSquare error for different polynome's degree
rmsTrainError = []
rmsTestError = []
degree = []
for n in range(10):
    degree.append(n)
    N = len(Xtrain)
    Nprime = len(Xtest)
    # Compute curve_coefficient and error on the training set, also print the training error
    curve_coefficient, error = lin_reg(Xtrain, Ytrain, n)
    rms_error = np.sqrt(2*error/N)
    rmsTrainError.append(rms_error)
    #####

    # Compute and print error on the testing set
    testError = compute_testError(Xtest, Ytest, n, curve_coefficient)
    rms_testError = np.sqrt(2*testError/Nprime)
    rmsTestError.append(rms_testError)
    #####

# Compute linear regression with M = 9, and a regularization parameter
# According the random selection of the test dataset, I have more or less a figure like the one
# in the textbook.
M = 9
lbda = [1*10**i for i in range(-15,1)]
Erms_training = []
Erms_test = []
for l in lbda:
    curve_coefficient, error = lin_reg(Xtrain, Ytrain, M, l)
    Erms_training.append(np.sqrt(2*error/len(Xtrain)))
    testError = compute_testError(Xtest, Ytest, M, curve_coefficient)
    Erms_test.append(np.sqrt(2*testError/len(Xtest)))

# Plot figure
figure = plt.figure()
ax = plt.axes()
#
plt.subplot(221)
plt.title('Linear Regression with n = 3')
plt.plot(Xtrain, Ytrain, 'r+')
plt.plot(Abs, Yreg)
plt.plot(Xtest, Ytest, 'g.')
#
plt.subplot(222)
plt.title('Root-Mean-Squared error according to the polynome\'s degree')
plt.plot(degree,rmsTrainError, linestyle='-', marker='o')
plt.plot(degree,rmsTestError, linestyle='-', marker='o')
#
plt.subplot(223)
plt.title('Mean-square-root error versus ln(lambda)')
plt.plot(np.log(lbda),Erms_training)
plt.plot(np.log(lbda), Erms_test)
plt.show()
#####
