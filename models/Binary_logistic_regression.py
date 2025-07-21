class BinaryLogisticRegression:
    def __init__(self 
                , lr = 0.01 , n_iter=1000 , epoch=5 , batch_size=32
                , method = "BGD"):
        
        self.lr = lr
        self.n_iter = n_iter
        self.epoch = epoch
        self.batch_size = batch_size
        self.method = method
        
        self.w = None
        self.b = None
        
    # static method to check the dimension
    @staticmethod
    def check_ndim(x):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        return x
    
    # sigmoid function
    @staticmethod    
    def sigmoid(z):
        return 1/(1+np.exp(-z)) # returns vector/matrix of probabilities 
    
    # fit function
    def fit(self , x_train , y_train):
        x_train = BinaryLogisticRegression.check_ndim(x_train)
        n_samples , n_features = x_train.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        if self.method == "BGD":
            self.Batch_gradient_descent(self.lr , self.n_iter
                                       , x_train , y_train
                                       , n_samples)
        elif self.method == "SGD":
            self.Stochastic_gradient_descent(self.lr , self.epoch
                                   , x_train , y_train
                                   , n_samples)
        elif self.method == "MBGD":
            self.Mini_batch_gradient_descent(self.lr , self.epoch , self.batch_size
                                            , x_train , y_train , n_samples)
        
    def predict_proba(self , x_test):
        z = np.dot(x_test, self.w) + self.b
        return self.sigmoid(z)
    
    def predict(self , x_test):
        # find the probability of point/s belong to class 1 
        y_probs = self.predict_proba(x_test)
        
        # return 1 if it belongs to the class else return 0
        return np.where(y_probs >= 0.5, 1, 0)
    
    
    def Batch_gradient_descent(self 
                               , lr , n_iter
                               , x_train , y_train 
                               , n_samples):
        for i in range(n_iter):
            # prediction
            z = np.dot(x_train , self.w) + self.b
            
            y_pred = self.sigmoid(z)
            
            dw = (1/n_samples) * np.dot(x_train.T , (y_pred - y_train))
            db = (1/n_samples) * np.sum(y_pred - y_train)
            
            self.w = self.w - lr * dw
            self.b = self.b - lr * db
    
    def Stochastic_gradient_descent(self
                                   , lr , epoch
                                   , x_train , y_train
                                   , n_samples):
        index_shuffle = len(x_train)
        for ep in range(epoch):
            indices = np.random.permutation(index_shuffle)
            for index in indices:
                x = x_train[index].reshape(1,-1)
                y = y_train[index]
                
                z = np.dot(x , self.w) + self.b
                y_pred = self.sigmoid(z)
                
                dw = np.dot(x.T , (y_pred - y))
                db = np.sum(y_pred - y)
                
                self.w = self.w - lr * dw
                self.b = self.b - lr * db
    
    def Mini_batch_gradient_descent(self
                                   , lr , epoch , batch_size
                                   , x_train , y_train
                                   , n_samples):
        for ep in range(epoch):
            indices = np.random.permutation(n_samples)
            x_batch = x_train[indices]
            y_batch = y_train[indices]
            
            for start in range(0 , n_samples , batch_size):
                end = start + batch_size
                x = x_batch[start:end]
                y = y_batch[start:end]
                
                batch_len = y.shape[0]
                
                z = np.dot(x , self.w) + self.b
                y_pred = self.sigmoid(z)
                
                dw = (1/batch_len) * np.dot(x.T , (y_pred - y))
                db = (1/batch_len) * np.sum(y_pred - y)
                
                self.w = self.w - lr * dw
                self.b = self.b - lr * db