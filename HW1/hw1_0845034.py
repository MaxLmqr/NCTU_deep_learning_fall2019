import random
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

filename = { 'EnergyEfficiency': 'EnergyEfficiency_data.csv',
             'ionosphere': 'ionosphere_csv.csv'}

data_EE = open(filename['EnergyEfficiency'],'r').read().splitlines()

"""
0 : Relative compactness
1 : Surface Area
2 : Wall Area
3 : Roof Area
4 : Overall Height
5 : Orientation                 --> One-hot vector  
6 : Glazing Area    # 9 after one-hot vector
7 : Glazing Area Distribution   --> One-hot vector  # 10
8 : Heating Load
9 : Cooling Load
"""

############### LOAD DATA ######################################


data_EE = data_EE[1:]   # Remove headers
heating_load = []   # Separate the feature we want to guess afterwards


for i in range(len(data_EE)):
    data_EE[i] = data_EE[i].split(',')
    for j in range(8):
        data_EE[i][j] = float(data_EE[i][j])

    # one-hot vectorization for orientation and glazing distribution area
    data_EE[i][5] = [int(data_EE[i][5]==(j+2)) for j in range(4)]
    data_EE[i][7] = [int(data_EE[i][7]==(j+1)) for j in range(5)]
    
    # fill heating datas
    heating_load.append(float(data_EE[i][8]))

    # keep only the 8 features
    data_EE[i] = data_EE[i][:8]

    # process to have one vector which will have a length equal to 15
    temp = []
    for x in data_EE[i]:
        if type(x) is float:
            temp += [x]
        else:
            temp += x
    data_EE[i] = temp

# Shuffle data
ndata = len(data_EE)
indexes = [i for i in range(ndata)]
random.shuffle(indexes)
training_indexes = indexes[:math.floor(ndata*0.75)]
test_indexes = indexes[math.floor(ndata*0.75):]
training_indexes.sort()
test_indexes.sort()

train_data_EE = np.asarray([data_EE[i] for i in training_indexes])
test_data_EE = np.asarray([data_EE[i] for i in test_indexes])

train_heating = np.asarray([heating_load[i] for i in training_indexes])
test_heating = np.asarray([heating_load[i] for i in test_indexes])

train_data_EE = train_data_EE.T
test_data_EE = test_data_EE.T
train_heating = train_heating.reshape((-1,1))
test_heating = test_heating.reshape((-1,1))

train_data_norm = train_data_EE
test_data_norm = test_data_EE
for i in range(train_data_EE.shape[0]):
    train_data_norm[i,:] = (train_data_norm[i,:] - min(train_data_norm[i,:]))/(max(train_data_norm[i,:])-min(train_data_norm[i,:]))
    test_data_norm[i,:] = (test_data_norm[i,:] - min(test_data_norm[i,:]))/(max(test_data_norm[i,:])-min(test_data_norm[i,:]))
train_heating_norm = (train_heating-min(train_heating))/(max(train_heating)-min(train_heating))
test_heating_norm = (test_heating-min(test_heating))/(max(test_heating)-min(test_heating))

############################################################
################## NEURON SCHEMA ###########################

"""

First layer will be a 15 features input
    Dimension : 20x15 matrix of coefficient

Second layer will be 20 features
    Dimension : 10x20 matrix of coefficient

Third layer will be 10 features 
    Dimension : 1x10 matrix of coefficient

Fourth layer will be 1 feature output

"""

############################################################

def relu(z):
    R = np.maximum(0.01*z,z)
    return R


def relu_derivative(z):
    R = np.array(z, copy=True)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i][j]<=0:
                R[i][j] = 0.01
            else:
                R[i][j] = 1
    return R

# def softmax(z):
#     S = np.exp(z)
#     S = S / np.sum(S)
#     return S

# def sigmoid(z):
#     S = 1/(1+np.exp(-z))
#     return S

# def sigmoid_derivative(z):
#     S = sigmoid(z)*(1-sigmoid(z))
#     return S


# Renvoie la prédiction, ainsi que toutes les 'prédictions' intermédiaires
# Avant et après activation de la fonction
# Z is the result of the multiplication of X by the coefficient of the layer + bias, before activation
# H is the activation of Z
def forward_propagation(X,coefficient, bias):
    forward_propagation_cache = {0:[X,X]}
    H = X.reshape((X.shape[0],-1))
    n_layer = len(coefficient)
    for i in range(1,n_layer+1):
        new_X = H
        Z = (coefficient['W'+str(i)]@new_X)+bias['B'+str(i)]
        H = relu(Z)
        forward_propagation_cache[i] = [Z,H]
    return H, forward_propagation_cache


# Cost function, return the actual MSE
def cost_function(X,Y):
    return (X-Y.T)@(X-Y.T).T


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
            dC_dZ = np.multiply((H-y.T),relu_derivative(Z))
        else:
            # print(coefficient['W'+str(j+1)].shape,dC_dZ_prev.shape)
            dC_dZ = np.multiply((coefficient['W'+str(j+1)].T@dC_dZ_prev),relu_derivative(Z))

        dC_dZ_prev = dC_dZ

        # COMPUTE dZ/dw (j-th layer)
        dZ_dW = H_prev.T

        # COMPUTE Grad
        grad_coefficient = dC_dZ@dZ_dW
        grad_bias = np.sum(dC_dZ.reshape((dC_dZ.shape[0],-1)),axis=1)
        grad_bias = grad_bias.reshape(grad_bias.shape[0],-1)
        try:
            gradient['W'+str(j)] += grad_coefficient/n
            gradient['B'+str(j)] += grad_bias/n
        except:
            gradient['W'+str(j)] = grad_coefficient/n
            gradient['B'+str(j)] = grad_bias/n   
    return gradient


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
    coefficient = { 'W1':np.random.rand(20,first_layer_size)*1e-1,
                    'W2':np.random.rand(10,20)*1e-1,
                    'W3':np.random.rand(1,10)*1e-1
                    }
    bias = {'B1':np.zeros((20,1)),
            'B2':np.zeros((10,1)),
            'B3':np.zeros((1,1))
            }
    n_layer = 3
    n_examples = X.shape[1]

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        

        # Forward
        H, forward_cache = forward_propagation(X, coefficient, bias)
        
        # Compute cost
        cost = cost_function(H, Y)
    
        # Backward
        grads = backward_propagation(H, Y, coefficient, forward_cache, n_layer)

        # # Update coefficient.
        coefficient, bias = update_coefficient(coefficient, bias, grads, learning_rate, n_layer)


        if i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))

        costs.append(cost)
    
    return coefficient,bias, costs


def RMS_error(X,Y):
    return (1/len(X)*((X-Y)@(X-Y).T))**0.5

def one_feature_deletion(X,Y,learning_rate, num_iterations):
    n = X.shape[0]
    last_cost = []
    for i in tqdm(range(n)):
        X_reduced = np.delete(X,(i),axis=0)
        W,b, costs = NN_model(X, Y, learning_rate,num_iterations)
        last_cost.append(costs[len(costs)-1])
    return last_cost.index(max(last_cost))

def multiple_feature_deletion(train_data,Y,learning_rate, num_iterations, num_features):
    X = np.copy(train_data)
    for i in tqdm(range(num_features)):
        index_to_delete = one_feature_deletion(X,Y,learning_rate,num_iterations)
        X = np.delete(X,(index_to_delete),axis=0)
    return X


######################### COMPUTE RESULTS #######################################################
num_iterations = 1000
learning_rate = 0.2
selection_feature = 0
W,b, costs = NN_model(train_data_norm,train_heating_norm, learning_rate,num_iterations)

###### Feature Selection
if selection_feature == 1:
    train_data_norm = np.delete(train_data_norm,(8,9,10,11,12,13,14),axis=0)
    test_data_norm = np.delete(test_data_norm,(8,9,10,11,12,13,14),axis=0)
    W,b, costs = NN_model(train_data_norm,train_heating_norm, learning_rate,num_iterations)


###### TRAINING CURVE
X_training_curve = [i for i in range(num_iterations)]
assert(len(costs) == num_iterations)
costs = [x[0][0] for x in costs]

###### TRAIN PREDICTION
Ytrain = []
for i in range(train_data_norm.shape[1]):
    h,cac = forward_propagation(train_data_norm[:,i],W,b)
    Ytrain.append(h)
Ytrain = [x[0][0] for x in Ytrain]
train_heating_norm = [x[0] for x in train_heating_norm]
X_train_display = [i for i in range(train_data_EE.shape[1])]
train_RMS_error = RMS_error(np.asarray(Ytrain),np.asarray(train_heating_norm))


###### TEST PREDICTION
Ypredicted = []
for i in range(test_data_norm.shape[1]):
    h,cac = forward_propagation(test_data_norm[:,i],W,b)
    Ypredicted.append(h)
Ypredicted = [x[0][0] for x in Ypredicted]
test_heating_norm = [x[0] for x in test_heating_norm]
X_test_display  = [i for i in range(test_data_EE.shape[1])]
test_RMS_error = RMS_error(np.asarray(Ypredicted),np.asarray(test_heating_norm))


##### TABLE DATA, with Network Architecture, Train and Test RMS
table_data = [['Network Architecture', '15-20-10-1'],
                ['Training RMS', train_RMS_error],
                ['Test RMS', test_RMS_error]]


################################### DISPLAY ########################################################
fig, ax = plt.subplots(nrows=2, ncols=2)
plt.tight_layout(1.5)


ax[0][0].axis('tight')
ax[0][0].axis('off')
table = ax[0][0].table(cellText=table_data, loc='center', fontsize=32)
table.auto_set_font_size(False)
table.set_fontsize(16)

ax[0][1].set_title('Training curve')
ax[0][1].plot(X_training_curve,costs)

ax[1][0].set_title('Prediction for training data')
ax[1][0].plot(X_train_display,Ytrain,label='Predict')
ax[1][0].plot(X_train_display,train_heating_norm,label='Label')
ax[1][0].legend(loc='upper left')


ax[1][1].set_title('Prediction for test set')
ax[1][1].plot(X_test_display,Ypredicted,label='Predict')
ax[1][1].plot(X_test_display,test_heating_norm,label='Label')
ax[1][1].legend(loc='upper left')


plt.show()