class OVA:
    """
    One-vs-All (OvA) multiclass classifier using Logistic Regression.
    
    Supports three optimization methods: BGD, SGD, and MBGD.
    """

    def __init__(self, lr=0.01, n_iter=1000, epoch=5, batch_size=32, method="BGD"):
        """
        Initialize OVA classifier.

        Parameters:
        - lr (float): Learning rate.
        - n_iter (int): Number of iterations for BGD.
        - epoch (int): Number of epochs for SGD/MBGD.
        - batch_size (int): Mini-batch size.
        - method (str): Optimization method ("BGD", "SGD", "MBGD").
        """
        self.lr = lr
        self.n_iter = n_iter
        self.epoch = epoch
        self.batch_size = batch_size
        self.method = method

        self.unique_classes = None  # All unique classes in training labels
        self.Classifiers = {}       # Dict: class -> (weights, bias)

    @staticmethod
    def check_ndim(x):
        """
        Check if input has 1D shape.

        Returns:
        - True if x.ndim == 1, else False.
        """
        return x.ndim == 1

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function.

        Parameters:
        - z (np.ndarray): Linear combination of features.

        Returns:
        - Sigmoid probabilities.
        """
        return 1 / (1 + np.exp(-z))

    def binary_classification(self, x, y):
        """
        Train binary logistic regression classifier.

        Parameters:
        - x (np.ndarray): Training features.
        - y (np.ndarray): Binary labels (0 or 1).

        Returns:
        - (w, b): Trained weights and bias.
        """
        if self.check_ndim(x):
            x = x.reshape(-1, 1)

        n_samples, n_features = x.shape
        w = np.zeros(n_features)
        b = 0

        if self.method == "BGD":
            w, b = self.Batch_gradient_descent(self.lr, self.n_iter, x, y, n_samples, w, b)
        elif self.method == "SGD":
            w, b = self.Stochastic_gradient_descent(self.lr, self.epoch, x, y, n_samples, w, b)
        elif self.method == "MBGD":
            w, b = self.Mini_batch_gradient_descent(self.lr, self.epoch, self.batch_size, x, y, n_samples, w, b)

        return w, b

    def fit(self, x_train, y_train):
        """
        Fit OvA model on training data.

        Parameters:
        - x_train (np.ndarray): Training features.
        - y_train (np.ndarray): Multiclass labels.
        """
        self.unique_classes = np.unique(y_train)

        for cls in self.unique_classes:
            binary_y = np.where(cls == y_train, 1, 0)
            w, b = self.binary_classification(x_train, binary_y)
            self.Classifiers[cls] = (w, b)

    def predict_proba(self, x_test):
        """
        Predict class probabilities for each class.

        Parameters:
        - x_test (np.ndarray): Test features.

        Returns:
        - probas (np.ndarray): Shape (n_samples, n_classes)
        """
        probas = []
        for cls in self.unique_classes:
            w, b = self.Classifiers[cls]
            z = np.dot(x_test, w) + b
            p = self.sigmoid(z)
            probas.append(p)

        return np.array(probas).T

    def predict(self, x_test):
        """
        Predict the class label for each sample.

        Parameters:
        - x_test (np.ndarray): Test features.

        Returns:
        - y_pred (np.ndarray): Predicted class labels.
        """
        probas = self.predict_proba(x_test)
        return self.unique_classes[np.argmax(probas, axis=1)]

    def Batch_gradient_descent(self, lr, n_iter, x_train, y_train, n_samples, w, b):
        """
        Perform Batch Gradient Descent.
        """
        for _ in range(n_iter):
            z = np.dot(x_train, w) + b
            y_pred = self.sigmoid(z)

            dw = (1 / n_samples) * np.dot(x_train.T, (y_pred - y_train))
            db = (1 / n_samples) * np.sum(y_pred - y_train)

            w -= lr * dw
            b -= lr * db

        return w, b

    def Stochastic_gradient_descent(self, lr, epoch, x_train, y_train, n_samples, w, b):
        """
        Perform Stochastic Gradient Descent.
        """
        for _ in range(epoch):
            indices = np.random.permutation(n_samples)
            for index in indices:
                x = x_train[index].reshape(1, -1)
                y = y_train[index]

                z = np.dot(x, w) + b
                y_pred = self.sigmoid(z)

                dw = np.dot(x.T, (y_pred - y))
                db = np.sum(y_pred - y)

                w -= lr * dw
                b -= lr * db

        return w, b

    def Mini_batch_gradient_descent(self, lr, epoch, batch_size, x_train, y_train, n_samples, w, b):
        """
        Perform Mini-Batch Gradient Descent.
        """
        for _ in range(epoch):
            indices = np.random.permutation(n_samples)
            x_batch = x_train[indices]
            y_batch = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                x = x_batch[start:end]
                y = y_batch[start:end]

                z = np.dot(x, w) + b
                y_pred = self.sigmoid(z)

                dw = (1 / len(y)) * np.dot(x.T, (y_pred - y))
                db = (1 / len(y)) * np.sum(y_pred - y)

                w -= lr * dw
                b -= lr * db

        return w, b
