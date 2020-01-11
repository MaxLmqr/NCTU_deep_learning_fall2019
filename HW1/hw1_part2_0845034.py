import numpy as np
import random
import math
import matplotlib.pyplot as plt
np.random.seed(3)
random.seed(3)
data = open('ionosphere_csv.csv','r').read().splitlines()
data = data[1:]
label = []
number_feature = 34

for i in range(len(data)):
    data[i] = data[i].split(',')
    label.append(1 if data[i][34]=='g' else 0)
    data[i] = data[i][:number_feature]
    for j in range(number_feature):   
        try:  
            data[i][j] = float(data[i][j])
        except:
            raise Exception('The data is not a number')

ndata = len(data)
indexes = [i for i in range(ndata)]
random.shuffle(indexes)
training_indexes = indexes[:math.floor(ndata*0.8)]
test_indexes = indexes[math.floor(ndata*0.8):]
training_indexes.sort()
test_indexes.sort()

train_data = np.asarray([data[i] for i in training_indexes])
test_data = np.asarray([data[i] for i in test_indexes])
train_label = np.asarray([label[i] for i in training_indexes])
test_label = np.asarray([label[i] for i in test_indexes])

train_data = train_data.T
test_data = test_data.T
train_label = train_label.reshape((-1,1))
test_label = test_label.reshape((-1,1))

################### NN Model #############################

def sigmoid(z):
    S = 1/(1+np.exp(-z))
    return S

def sigmoid_derivative(z):
    S = sigmoid(z)*(1-sigmoid(z))
    return S

def forward_propagation(X,coefficient, bias):
    forward_propagation_cache = {0:[X,X]}
    H = X.reshape((X.shape[0],-1))
    n_layer = len(coefficient)
    for i in range(1,n_layer+1):
        new_X = H
        Z = (coefficient['W'+str(i)]@new_X)+bias['B'+str(i)]
        H = sigmoid(Z)
        forward_propagation_cache[i] = [Z,H]
    return H, forward_propagation_cache

def backward_propagation(X,Y,coefficient,forward_propagation_cache, layer_num):
    # Forward_propagation_cache contient pour chaque layer le vecteur avant/après activation
    # On va créer un vecteur gradient contenant les dérivées par rapports aux paramètres et au bias
    gradient = {}
    n = X.shape[1]
    # Init
    y = Y
    dC_dZ_prev = 0
    for j in reversed(range(1,layer_num+1)):
        Z,H = forward_propagation_cache[j]
        H_prev = forward_propagation_cache[j-1][1]

        

        # COMPUTE dC/dz (j-th layer)
        if j==layer_num:
            temp = -(np.divide(y.T,H)-np.divide(1-y.T,1-H))
            dC_dZ = np.multiply((temp),sigmoid_derivative(Z))
        else:
            # print(coefficient['W'+str(j+1)].shape,dC_dZ_prev.shape)
            dC_dZ = np.multiply((coefficient['W'+str(j+1)].T@dC_dZ_prev),sigmoid_derivative(Z))

        dC_dZ_prev = dC_dZ

        # COMPUTE dZ/dw (j-th layer)
        dZ_dW = H_prev.T

        # COMPUTE Grad
        grad_coefficient = dC_dZ@dZ_dW
        grad_bias = np.sum(dC_dZ.reshape((dC_dZ.shape[0],-1)),axis=1)
        grad_bias = grad_bias.reshape(grad_bias.shape[0],-1)
        try:
            gradient['W'+str(j)] += grad_coefficient
            gradient['B'+str(j)] += grad_bias
        except:
            gradient['W'+str(j)] = grad_coefficient
            gradient['B'+str(j)] = grad_bias
    gradient = {k: v/n for k, v in gradient.items()}
    return gradient



def cost_function(X,Y):
    # X vecteur ligne
    total_error = 0
    for i in range(len(Y)):
        if Y[i] == 1:
            total_error += -np.log(X[0][i])
        else:
            total_error += -np.log(1-X[0][i])
    return total_error

    

def update_coefficient(coefficient, bias, grads, learning_rate, n_layer):
    for i in range(1,n_layer+1):
        coefficient['W'+str(i)] = coefficient['W'+str(i)] - learning_rate*grads['W'+str(i)]
        bias['B'+str(i)] = bias['B'+str(i)] - learning_rate*grads['B'+str(i)]
    return coefficient, bias

def NN_model(X, Y, learning_rate = 0.1, num_iterations = 1000):

    # keep track of cost
    costs = []
    first_layer_size = X.shape[0]
    # Define random coefficient abd bias (INITIALIZATION)
    coefficient = { 'W1':np.random.rand(6,first_layer_size)*1e-0,
                    'W2':np.random.rand(2,6)*1e-0,
                    'W3':np.random.rand(1,2)*1e-0
                    }
    bias = {'B1':np.zeros((6,1)),
            'B2':np.zeros((2,1)),
            'B3':np.zeros((1,1)),
            }
    n_layer = 3
    n_examples = X.shape[1]

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        

        # Forward propagation
        H, forward_cache = forward_propagation(X, coefficient, bias)
        
        # Compute cost.
        cost = cost_function(H, Y)
    
        # Backward propagation.
        grads = backward_propagation(H, Y, coefficient, forward_cache, n_layer)

        # # Update coefficient.
        coefficient, bias = update_coefficient(coefficient, bias, grads, learning_rate, n_layer)
        # Print the cost every 100 training example
        if i % 1000 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            

        if i==10:
            W10 = coefficient.copy()
            b10 = bias.copy()
        
        if i==1000:
            W1000 = coefficient.copy()
            b1000 = bias.copy()



        costs.append(cost)
    
    return coefficient,bias, costs, W10, b10, W1000, b1000

def compute_error(X,Y):
    error = 0
    for i in range(len(X)):
        if Y[i] == 1:
            error += -np.log(X[i])
        else:
            error += -np.log(1-X[i])
    error = error/len(X)
    return error

######################### COMPUTE RESULTS #######################
##################################################################


train_error = 0
num_iterations = 7000
learning_rate= 0.08
threshold = 0.4



W,b,costs, W10, b10, W1000, b1000 = NN_model(train_data,train_label,learning_rate,num_iterations)

Xcost = [i for i in range(num_iterations)]
plt.figure()
plt.plot(Xcost,costs)
# w10 = np.sum([np.sum(np.square(W10['W'+str(i)])) for i in range(1,4)])
# w1000 = np.sum([np.sum(np.square(W1000['W'+str(i)])) for i in range(1,4)])
# w = np.sum([np.sum(np.square(W['W'+str(i)])) for i in range(1,4)])
# plt.plot([10,1000,7000],[w10,w1000,w])
plt.show()

Ytrain = []
train_cac= []
Ytrain10 = []
Ytrain1000 = []
for i in range(train_data.shape[1]):
    h,cac = forward_propagation(train_data[:,i],W,b)
    h10,cac10 = forward_propagation(train_data[:,i],W10,b10)
    h1000,cac1000 = forward_propagation(train_data[:,i],W1000,b1000)
    Ytrain.append(h)
    train_cac.append(cac)
    Ytrain10.append(h10)
    Ytrain1000.append(h1000)
Ytrain = [x[0][0] for x in Ytrain]
Ytrain10 = [x[0][0] for x in Ytrain10]
Ytrain1000 = [x[0][0] for x in Ytrain1000]
train_label = [x[0] for x in train_label]
train_error = compute_error(Ytrain,train_label)
print('Cost of the training set : ',train_error)
trainPrediction  = [1 if Ytrain[i]>=threshold else 0 for i in range(len(Ytrain))]
wrong_prediction = 0
for k in range(len(trainPrediction)):
    if trainPrediction[k] != train_label[k]:
        wrong_prediction += 1
wrong_prediction = (wrong_prediction/len(trainPrediction))*100
print('Error rate for the training set : ',wrong_prediction)


Ytest = []
for i in range(test_data.shape[1]):
    h,cac = forward_propagation(test_data[:,i],W,b)
    Ytest.append(h)
Ytest = [x[0][0] for x in Ytest]
test_label = [x[0] for x in test_label]
test_error = compute_error(Ytest,test_label)
print('Cost of the test set : ',test_error)

testPrediction  = [1 if Ytest[i]>=threshold else 0 for i in range(len(Ytest))]
wrong_prediction = 0
for k in range(len(testPrediction)):
    if testPrediction[k] != test_label[k]:
        wrong_prediction += 1
wrong_prediction = wrong_prediction/len(testPrediction)*100
print('Test error rate : ', wrong_prediction)


# train_data = test_data
# Ytrain = Ytest
# h,cache = forward_propagation(train_data,W10,b10)
# class1_index = [i for i in range(len(Ytrain)) if Ytrain[i]>=threshold]
# class2_index = [i for i in range(len(Ytrain)) if Ytrain[i]<threshold]
# X_disp = cache[2][1][0]
# Y_disp = cache[2][1][1]
# X_disp_class1 = [X_disp[i] for i in class1_index]
# X_disp_class2 = [X_disp[i] for i in class2_index]
# Y_disp_class1 = [Y_disp[i] for i in class1_index]
# Y_disp_class2 = [Y_disp[i] for i in class2_index]
# plt.figure()
# plt.title('After 10 epochs')
# plt.plot(X_disp_class1,Y_disp_class1,'ro')
# plt.plot(X_disp_class2,Y_disp_class2,'go')


# h,cache = forward_propagation(train_data,W1000,b1000)
# class1_index = [i for i in range(len(Ytrain)) if Ytrain[i]>=0.5]
# class2_index = [i for i in range(len(Ytrain)) if Ytrain[i]<0.5]
# X_disp = cache[2][1][0]
# Y_disp = cache[2][1][1]
# X_disp_class1 = [X_disp[i] for i in class1_index]
# X_disp_class2 = [X_disp[i] for i in class2_index]
# Y_disp_class1 = [Y_disp[i] for i in class1_index]
# Y_disp_class2 = [Y_disp[i] for i in class2_index]

# plt.figure()
# plt.title('After 1000 epochs')
# plt.plot(X_disp_class1,Y_disp_class1,'ro')
# plt.plot(X_disp_class2,Y_disp_class2,'go')


h,cache = forward_propagation(train_data,W,b)
class1_index = [i for i in range(len(Ytrain)) if Ytrain[i]>=0.5]
class2_index = [i for i in range(len(Ytrain)) if Ytrain[i]<0.5]
X_disp = cache[2][1][0]
Y_disp = cache[2][1][1]
X_disp_class1 = [X_disp[i] for i in class1_index]
X_disp_class2 = [X_disp[i] for i in class2_index]
Y_disp_class1 = [Y_disp[i] for i in class1_index]
Y_disp_class2 = [Y_disp[i] for i in class2_index]
plt.figure()
plt.plot(X_disp_class1,Y_disp_class1,'ro')
plt.plot(X_disp_class2,Y_disp_class2,'go')
plt.show()
