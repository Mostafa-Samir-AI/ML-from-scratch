import numpy as np

class Regularization:
    def __init__(self , lr=0.01 , n_iter=1000 ,
                 epoch = 5 , fit_method = "BGD" ,
                 alpha = 1 ,
                 Regularization = True ,
                 Regularization_method="L1"):
        
        self.lr = lr
        self.n_iter = n_iter
        
        self.epoch = epoch
        self.fit_method = fit_method
        
        self.w = None
        self.b = None
        
        # regularization parameters
        self.Regularization = Regularization
        self.Regularization_method = Regularization_method
        self.alpha = alpha
        self.lasso_keys = ["L1" , "l1" , "lasso" , "Lasso"]
        self.ridge_keys = ["L2" , "l2" , "ridge" , "Ridge"]
    
    # static methods to assign penalty
    @staticmethod
    def ridge(n_samples , alpha , w):
        return (alpha/n_samples)*w 
    @staticmethod
    def lasso(n_samples , alpha , w):
        return (alpha/n_samples)*np.sign(w)
    
    # static method to check dimensions of input
    @staticmethod
    def check_dim(x):
        if x.ndim == 1:
            return True
        return False
    
    def fit(self , x_train , y_train):
        
        # check the dimension of the inputs
        if self.check_dim(x_train):
            x_train = x_train.reshape(-1,1)
            n_samples = x_train.shape[0]
            n_features = 1
        else:
            n_samples , n_features = x_train.shape
        
        # initialize dimensions of inputs
        self.w = np.zeros(n_features)
        self.b = 0
        
        # fitting
        if self.fit_method == "BGD":
            self.BGD(x_train , y_train 
                     , self.lr
                     , self.n_iter , n_samples 
                     , self.Regularization , self.Regularization_method , self.alpha)
        
        elif self.fit_method == "SGD":
            self.SGD(x_train , y_train , self.epoch
           , self.lr
           , n_samples
           , self.Regularization , self.Regularization_method , self.alpha)
        
        elif self.fit_method == "MBGD":
            self.MBGD(x_train , y_train 
             , self.epoch , self.lr
             , n_samples , batch_size=32
             ,Regularization= self.Regularization ,Regularization_method= self.Regularization_method , alpha=self.alpha)
        
    
    def predict(self , x_test):
        if x_test.ndim == 1:
            # If it's a single sample with n_features
            if x_test.shape[0] == self.w.shape[0]:
                x_test = x_test.reshape(1, -1)  # shape (1, n_features)
            else:
                x_test = x_test.reshape(-1, 1)  # fallback: assume it's (n_samples, 1)
        
        return np.dot(x_test, self.w) + self.b
        
    def BGD(self , x_train , y_train 
            , lr 
            , n_iter , n_samples 
            , Regularization , Regularization_method , alpha):
        
        # container for the penalty function
#         penalty_func = pass
        
        if Regularization and Regularization_method in self.lasso_keys:
            penalty_func = self.lasso
        elif Regularization and Regularization_method in self.ridge_keys:
            penalty_func = self.ridge
        else:
            penalty_func = lambda n_samples, alpha, w: 0
            
        for i in range(n_iter):
            y_pred = np.dot(x_train , self.w) + self.b
            
            # update penalty
            penalty = penalty_func(n_samples , alpha , self.w)
            
            dw = (1/n_samples) * np.dot(x_train.T ,(y_pred - y_train)) + penalty
            db = (1/n_samples) * np.sum(y_pred - y_train)
            
            # update values + penalty
            self.w = self.w - lr * dw 
            self.b = self.b - lr * db
            
    def SGD(self , x_train , y_train , epoch
           , lr
           , n_samples
           , Regularization , Regularization_method , alpha):
        
        # condition for regularization penalty
        if Regularization and Regularization_method in self.lasso_keys:
            penalty_func = self.lasso
        elif Regularization and Regularization_method in self.ridge_keys:
            penalty_func = self.ridge
        else:
            penalty_func = lambda n_samples, alpha, w: 0
        
        # looping through epoch
        for epoch_ in range(epoch): 
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                x = x_train[i].flatten()
                y = y_train[i]
                
                y_pred = np.dot(x,self.w) + self.b
                
                # update penalty
                penalty = penalty_func(n_samples , alpha , self.w)

                dw = x *(y_pred - y) + penalty
                db = y_pred - y

                # update values + penalty
                self.w = self.w - lr * dw 
                self.b = self.b - lr * db
                
                
    def MBGD(self , x_train , y_train 
             , epoch , lr
             , n_samples , batch_size
             , Regularization , Regularization_method , alpha):
          
        if Regularization and Regularization_method in self.lasso_keys:
            penalty_func = self.lasso
        elif Regularization and Regularization_method in self.ridge_keys:
            penalty_func = self.ridge
        else:
            penalty_func = lambda n_samples, alpha, w: 0
        
        # making epochs
        for i in range(epoch):
            indices = np.random.permutation(n_samples)
            x_shuffle , y_shuffle = x_train[indices] , y_train[indices]
            
            for start in range(0 ,n_samples , batch_size):
                end = n_samples if end >= n_samples else start + batch_size
                
                x = x_shuffle[start:end]
                y = y_shuffle[start:end]
                
                y_pred = np.dot(x , self.w) + self.b
                
                penalty = penalty_func(n_samples , alpha , self.w)
                
                batch_len = x.shape[0]
                dw = (1/batch_len) * np.dot(x.T ,(y_pred - y)) + penalty
                db = (1/batch_len) * np.sum(y_pred - y)
                
                self.w = self.w - lr * dw
                self.b = self.b - lr * db