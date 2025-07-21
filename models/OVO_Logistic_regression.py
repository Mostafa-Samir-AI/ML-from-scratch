class OneVsOneLogisticRgeression:
    def __init__(self , lr =0.01
                ,n_iter = 1000
                ,epoch = 5 , batch_size=32
                , method = "BGD"):
        
        import numpy as np
        from collections import Counter 
        
        self.lr = lr
        self.n_iter = n_iter
        self.epoch = epoch
        self.batch_size = batch_size
        
        # paramters
        self.w = None
        self.b = None
        
        # defining method
        self.method = method
        
        # classifiers
        self.unique_classes = []
        self.classifiers = {}
        
    @staticmethod
    def check_ndim(x):
        if x.ndim == 1:
            return True
        return False
    
    @staticmethod
    def sigmoid(z):
        sig = 1/(1+np.exp(-z))
        return sig
    
    # optimization algorithms
    ## batch
    def Batch_gradient_descent(self 
                               , lr
                               , x_train , y_train
                               , w , b
                               , n_iter , n_samples):
        
        for i in range(n_iter):
            # prediction
            z = np.dot(x_train , w) + b
            y_pred = self.sigmoid(z)
            
            # update
            dw = (1/n_samples) * np.dot(x_train.T , (y_pred - y_train))
            db = (1/n_samples) * np.sum((y_pred - y_train))

            w = w - lr * dw
            b = b - lr * db
        return w , b
    
    # stochastic
    def Stochastic_gradient_descent(self
                                   , lr
                                   , x_train , y_train
                                   , w , b
                                   , epoch , n_samples):
        for ep in range(epoch): 
            # make an index to pick data points randomly
            random_index = np.random.permutation(n_samples)
            
            for index in random_index:
                x = x_train[index]
                y = y_train[index]
                
                z = np.dot(x , w) + b
                y_pred = self.sigmoid(z)
                
                dw = np.dot(x.T , (y_pred - y))
                db = np.sum(y_pred-y)
                
                w -= lr * dw
                b -= lr * dw
                
        return w , b
    
    # mini batch
    def Mini_batch_gradient_descent(self
                                   ,lr
                                   ,x_train,y_train
                                   ,w,b
                                   ,epoch,n_samples,batch_size):
        for ep in range(epoch):
            random_index = np.random.permutation(n_samples)
            x_shuffle = x_train[random_index]
            y_shuffle = y_train[random_index]
            
            
            for start in range(0 , n_samples , batch_size):
                end = min((start + batch_size),n_samples) 
                x = x_shuffle[start:end]
                y = y_shuffle0[start:end]
                
                z = np.dot(x , w) + b
                y_red = self.sigmoid(z)
                
                x_length = len(x)
                
                dw = (1/x_length) *np.dot(x.T , (y_pred - y))
                db = (1/x_length) * np.sum(y_pred-y)
                
                w -= lr * dw
                b -= lr * dw
                
        return w , b
    
    def binary_classification(self, x_train , y_train):
        if self.check_ndim(x_train):
            x_train = x_train.reshape(-1,1)

        n_samples , n_features = x_train.shape
        self.w = np.zeros(n_features)
        self.b = 0


        if self.method == "BGD":
            w , b = self.Batch_gradient_descent(self.lr
                                       ,x_train , y_train
                                       ,self.w , self.b
                                       ,self.n_iter , n_samples)
        elif self.method == "SGD":
            w , b = self.Stochastic_gradient_descent(self.lr
                                            ,x_train,y_train
                                            ,self.w,self.b
                                            ,self.epoch,n_samples)
        elif self.method == "MBGD":
            w , b = self.Mini_batch_gradient_descent(self.lr
                                            ,x_train,y_train
                                            ,self.w,self.b
                                            ,self.epoch,n_samples,self.batch_size)
        return w , b
    
    
    # fit function
    def fit(self , x_train , y_train):
        
        # get unique classes
        self.unique_classes = np.unique(y_train)
        
        for class_1 , class_2 in combinations(self.unique_classes , 2):
            # find all data points belong to class_1 and class_2
            index = np.where((y_train == class_1) | (y_train == class_2))
            
            x = x_train[index]
            y = y_train[index]
            
            binary_y = np.where(y == class_1 , 1 , 0)
            
            w , b = self.binary_classification(x , binary_y)
            
            self.classifiers[(class_1 , class_2)] = (w , b)
    
    # predict function
    def predict_proba(self , x_test):
        votes = []
        for (class_1 , class_2) , (w,b) in self.classifiers.items():
            z = np.dot(x_test,w) + b
            proba = self.sigmoid(z)
            pred = np.where(proba >0.5 , class_1 , class_2)
            votes.append(pred)
            
        votes = np.array(votes).T
        
        final_preds = []
        for sample in votes:
            most_common = Counter(sample).most_common(1)[0][0]
            final_preds.append(most_common)
        
        return np.array(final_preds)