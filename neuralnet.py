import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import utils



class NeuralNet: 
    params = {}
    cache = {}
    
    def __init__(self, input_size, neurons_num):
        np.random.seed(0)
        self.input_size = input_size
        self.neurons_num = neurons_num
        self.params = self.init(self.input_size, self.neurons_num)

    def init(self, input_size, neurons_numb):
        W0 = np.random.rand(neurons_numb,input_size)
        b0 = np.zeros((neurons_numb,1))
        W1 = np.random.rand(1,neurons_numb) 
        b1 = np.zeros((1,1))
        return W0,b0,W1,b1

    def train(self, X_train, Y_train, epoch_num, batch_size=1, video_progress = False, cost_fun='cross_entropy', lambd=0.01):
        counter = 0
        m = X_train.shape[1]
        batch_numb = math.floor(m/batch_size)
        costs = []
        
        for epoch in range(1,epoch_num+1):      # from 1 to epoch_num
            
            for j in range(batch_numb):    # go through batches
                x = X_train[:,j*batch_size:j*batch_size+batch_size]
                y = Y_train[:,j*batch_size:j*batch_size+batch_size]
                y_hat, self.cache = forward_pass(x,self.params)
                grads = back_pass(y, y_hat, cost_fun, self.cache)
                self.params = update_params(self.params, grads, lambd)

            if (m % batch_size) != 0:       # the last batch may be of different size
                x = X_train[:,batch_numb*batch_size:]
                y = Y_train[:,batch_numb*batch_size:]
                y_hat, self.cache = forward_pass(x,self.params)
                grads = back_pass(y, y_hat, cost_fun, self.cache)
                self.params = update_params(self.params, grads, lambd)

            costs.append(cost(y, y_hat, cost_fun))
            
            if video_progress:
                vid_dur = 5  # in seconds
                fps = 10
                if epoch % max(1,int(epoch_num/(vid_dur*fps))) == 0: # when to save a vid frame
                    counter += 1       
                    W0,b0,W1,b1 = self.params
                    utils.save_matrix_as_img(W0,'output/images/', "img_"+str(counter))

            print("epoch: " + str(epoch) + "   cost: ", cost(y, y_hat, cost_fun))

        if video_progress:            
            utils.generate_video('output/images/img_%d.png', 'output/output.mp4')

        plt.plot(costs)
        plt.show()
        print(costs[-1])
    
    
def linear(x,w,b):
    Z = np.dot(w,x)+b   
    return Z

def relu(z):
    a = np.array(z, copy=True)
    a[a<0]=0
    return a

def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a    

def linear_back(dZ,w,x):
    m = x.shape[1]
    dx = np.dot(w.T,dZ)
    dw = (1./m)*np.dot(dZ,x.T)
    db = (1./m)*np.sum(dZ, axis = 1, keepdims=True) 
    return dx,dw,db

def relu_back(da, a):
    dz = np.array(da, copy=True)
    #dz[a>0]=1 is not necessary,  1 * a so de facto a
    dz[a<=0]=0
    return dz

def sigmoid_back(da, a):
    dz = da*a*(1-a)
    return dz

def forward_pass(X, params):    
    W0,b0,W1,b1 = params    
    z1 = linear(X,W0,b0)
    cache = {}
    cache["X"] = X
    cache["W0"] = W0
    cache["b0"] = b0
    cache["z1"] = z1 
    a1 = relu(z1)
    cache["a1"] = a1
    cache["W1"] = W1    
    z2 = linear(a1,W1,b1)
    cache["z2"] = z2    
    y_hat = sigmoid(z2)    
    return y_hat, cache

def back_pass(y, y_hat, cost_fun, cache):  
    m = y.shape[1]
    if cost_fun == 'cross_entropy': 
        dy_hat = (1./m)*(-(y/y_hat)+((1-y)/(1-y_hat)) )
    if cost_fun == 'quadratic':
        dy_hat = (1./m)*(y_hat - y)
    dz2 = sigmoid_back(dy_hat, y_hat) 
    da1, dw1, db1 = linear_back(dz2,cache["W1"],cache["a1"])
    dz1 = relu_back(da1,cache["a1"])
    _, dw0, db0 = linear_back(dz1,cache["W0"],cache["X"])    
    return dw0, db0, dw1, db1

def update_params(params, grads, lambd):
    W0,b0,W1,b1 = params
    dw0, db0, dw1, db1 = grads
    W0 = W0 - lambd*dw0
    b0 = b0 - lambd*db0
    W1 = W1 - lambd*dw1
    b1 = b1 - lambd*db1
    return W0,b0,W1,b1

def cost(y,y_hat, cost_fun):
    m = y.shape[1]
    if cost_fun == 'cross_entropy':
        cost = -(1./m)*np.sum( np.dot(y,np.log(y_hat).T) + np.dot(1-y,np.log(1-y_hat).T) )
    if cost_fun == 'quadratic':
        cost = (1./(2*m))*np.sum(np.power(y_hat-y,2))
    return cost
