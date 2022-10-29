import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(s):
    return s*(1-s)


class NeuralNetwork():
    def __init__(self, lr, n_iters):
        #parameters
        self.lr = lr
        self.n_iters = n_iters
        self.weight1 = None
        self.bias1 = None
        self.weight2 = None
        self.bias2 =None
        
    def fit(self, X, y):
        
        nrows,ncol=X.shape

        self.weight1=np.random.rand(ncol,nrows)
        self.weight2=np.random.rand(nrows,1)
        self.bias1=np.random.rand(1)
        self.bias2=np.random.rand(1)
        

        for count in range(self.n_iters):
                       
            L1=np.dot(X,self.weight1)+ self.bias1 #input x weight1 + bias1
            Activated1=sigmoid(L1)#activated function 1

            L2=np.dot(Activated1,self.weight2)+ self.bias2 #activated function 1 x weight2 + bias2
            output=sigmoid(L2)# activated function 2(output)

            output_error = y - output #actual result - output  
            
            # if ( count % 1000) == 0:
            #     print ("absolute mean Error: \n", str(np.mean(np.abs(output_error))),"\n")
                
            output_delta = output_error * sigmoid_deriv(output) 
            activated1_error = output_delta.dot(self.weight2.T)
            activated1_delta = activated1_error * sigmoid_deriv(Activated1)
            
            self.weight2 += Activated1.T.dot(output_delta) * self.lr
            self.weight1 += X.T.dot(activated1_delta) * self.lr
            self.bias2 += np.sum(output_error) * self.lr
            self.bias1 += np.sum(activated1_error) * self.lr
            
    def predict(self, X):
        L1=np.dot(X,self.weight1)+ self.bias1 #input x weight1
        Activated1=sigmoid(L1) #activated function 1
        L2=np.dot(Activated1,self.weight2)+ self.bias2 #activated function 1 x weight2
        output=sigmoid(L2) # activated function 2
        return output   