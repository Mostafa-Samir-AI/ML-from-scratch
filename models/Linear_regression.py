import numpy as np 
# import pandas as pd
import matplotlib.pyplot as plt

class LG:
    """
    Linear Regression model trained using Gradient Descent methods (BGD, SGD, MBGD).
    
    Supports tracking loss per step or epoch and visualizing loss curves.

    Parameters
    ----------
    lr : float, optional (default=0.01)
        Learning rate for gradient descent.
    n_iter : int, optional (default=1000)
        Number of iterations (used for BGD).
    epoch : int, optional (default=5)
        Number of epochs (used for SGD and MBGD).
    batch_size : int, optional (default=32)
        Batch size (used for MBGD).
    method : str, optional (default="BGD")
        Optimization method: "BGD", "SGD", or "MBGD".
    track_loss : str, optional (default="MSE")
        Loss metric to track: "MAE", "MSE", or "RMSE".
    """
    
    def __init__(self , lr=0.01 , n_iter=1000 , epoch= 5 , batch_size = 32, method = "BGD" , track_loss = "MSE"):
        import numpy as np
        import matplotlib.pyplot as plt
        
        self.lr = lr
        self.n_iter = n_iter
        self.epoch = epoch
        self.w = None
        self.b = None
        self.method = method
        self.batch_size = batch_size
        
        # track losses
        self.track_loss = track_loss
        self.loss_per_step = []
        self.loss_per_epoch = []
        
    @staticmethod
    def check_Dimension(x_train):
        """
        Check if the input data is one-dimensional.

        Parameters
        ----------
        x_train : ndarray
            Input features.

        Returns
        -------
        bool
            True if x_train is 1D, False otherwise.
        """
        if x_train.ndim == 1:
            return True
        return False

    
    def fit(self , x_train , y_train):
        """
        Train the model using the specified gradient descent method.

        Parameters
        ----------
        x_train : ndarray
            Training feature data.
        y_train : ndarray
            Training target data.
        """
        # see the n_samples and n_features
        x_D = self.check_Dimension(x_train)
        if (x_D == True):
            x_train = x_train.reshape(-1, 1)
            n_samples = x_train.shape[0]
            n_features = 1
        else:
            n_samples , n_features = x_train.shape
        
        # init W and B values
        self.w = np.zeros(n_features)
        self.b = 0
        
        # fitting the line 
        if self.method == "BGD":
            self.BGD(x_train ,y_train ,self.n_iter , n_samples)
        elif self.method == "SGD":
            self.SGD(x_train ,y_train ,self.epoch , n_samples)
        elif self.method == "MBGD":
            self.MBGD(x_train , y_train , self.batch_size , self.epoch , n_samples)
        
    
        
    def BGD(self , x_train , y_train , n_iter , n_samples):
        """
        Perform Batch Gradient Descent optimization.

        Parameters
        ----------
        x_train : ndarray
            Training feature data.
        y_train : ndarray
            Training target data.
        n_iter : int
            Number of iterations.
        n_samples : int
            Number of samples.
        """
        for i in range(n_iter):
            # prediction
            y_pred = np.dot(x_train, self.w) + self.b            
            # optimization
            dw = (1/n_samples) * np.dot(x_train.T , (y_pred-y_train))
            db = (1/n_samples) * np.sum(y_pred-y_train)
            
            # update the values
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
    
    
    def SGD(self , x_train , y_train , epoch , n_samples):
        """
        Perform Stochastic Gradient Descent optimization.

        Parameters
        ----------
        x_train : ndarray
            Training feature data.
        y_train : ndarray
            Training target data.
        epoch : int
            Number of epochs.
        n_samples : int
            Number of samples.
        """
        index_shuffle = len(x_train)
        for i in range(epoch):
            indices = np.random.permutation(index_shuffle)
            for index in indices:
                x = x_train[index].reshape(1, -1)
                y = y_train[index]
                
                y_pred = np.dot(x , self.w) + self.b
                
                # optimization
                dw = (1/n_samples) * np.dot(x.T , (y_pred-y))
                db = (1/n_samples) * np.sum(y_pred-y)

                # update the values
                self.w = self.w - self.lr * dw
                self.b = self.b - self.lr * db
                
                # track losses per step
                loss = self.calculate_error(y_pred , y , criteria=self.track_loss)
                self.loss_per_step.append(loss)
        
            # track loss per epoch
            loss = self.calculate_error(y_pred , y , criteria=self.track_loss)
            self.loss_per_epoch.append(loss)

    
    
    def MBGD(self , x_train , y_train , batch_size , epoch , n_samples):
        """
        Perform Mini-Batch Gradient Descent optimization.

        Parameters
        ----------
        x_train : ndarray
            Training feature data.
        y_train : ndarray
            Training target data.
        batch_size : int
            Size of each mini-batch.
        epoch : int
            Number of epochs.
        n_samples : int
            Number of samples.
         """
        for i in range(epoch):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0 , n_samples , self.batch_size):
                end = start + self.batch_size
                if end >= n_samples:
                    end = n_samples
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = np.dot(x_batch, self.w) + self.b            
                # optimization
                dw = (1/n_samples) * np.dot(x_batch.T , (y_pred-y_batch))
                db = (1/n_samples) * np.sum(y_pred-y_batch)

                # track losses per step
                loss = self.calculate_error(y_pred , y_batch , criteria=self.track_loss)
                self.loss_per_step.append(loss)

            # track loss per epoch
            loss = self.calculate_error(y_pred , y_batch , criteria=self.track_loss)
            self.loss_per_epoch.append(loss)

    
    def predict(self , x_test):
        """
        Predict target values for test data.

        Parameters
        ----------
        x_test : ndarray
            Test feature data.

        Returns
        -------
        ndarray
            Predicted target values.
        """
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        return np.dot(x_test , self.w) + self.b
    
    
    def calculate_error(self , y_pred , y_test , criteria="MAE"):
        """
        Calculate the error between predictions and true values.

        Parameters
        ----------
        y_pred : ndarray
            Predicted values.
        y_test : ndarray
            Actual target values.
        criteria : str, optional (default="MAE")
            Type of error to calculate: "MAE", "MSE", or "RMSE".

        Returns
        -------
        float
            The calculated error.
        """
        if criteria == "MAE":
            return np.mean(np.abs(y_test - y_pred))
        elif criteria == "MSE":
            return np.mean((y_test - y_pred) ** 2)
        elif criteria == "RMSE":
            return np.sqrt(np.mean((y_test - y_pred) ** 2))
        
    
    def plot_loss_curve(self , loss_type = "step"):
        """
        Plot the loss curve over training updates.

        Parameters
        ----------
        loss_type : str, optional (default="step")
            Type of loss curve: "step" for per update or "epoch" for per epoch.

        Raises
        ------
        ValueError
            If loss_type is not "step" or "epoch".
        """
        if loss_type == "step":
            loss = self.loss_per_step
            title = f"{self.method} - Loss per Step"
            xlabel = "Step"
        elif loss_type == "epoch":
            loss = self.loss_per_epoch
            title = f"{self.method} - Loss per Epoch"
            xlabel = "Epoch"
        else:
            raise ValueError("loss_type must be either 'step' or 'epoch'")
        
        plt.figure(figsize=(10, 5))
        plt.plot(loss, label=f'{self.method} Loss Curve', color='blue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(self.track_loss)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()