import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from joblib import Parallel, delayed
import multiprocessing


class LogisticRegression:

    def __init__(self):
        print("Logistic Regression Class created")
        self.all_theta = []
        self.list_classes = []

    def fit(self, X, y, l):
        """Train Logistic Regression

        :param X: train data matrix. Columns represents features and rows instances)
        :param y: class labels use for training (lenght is equal to number of rows of X)
        :param l: parameter of regularization lambda
        """
        self.list_classes = list(set(y))
        self.list_classes.sort()

        classes = len(self.list_classes)
        X = self.add_theta0(X)
        num_cores = multiprocessing.cpu_count() -1
        results = Parallel(n_jobs=num_cores)(delayed(self.logistic_train_one_class)(X, y, self.list_classes, l, c)
                                             for c in range(classes))
        self.all_theta = np.asarray(results)

    def predict(self, X):
        """Predict labels for matrix data X

        :param X: data matrix
        :return: list of predicted labels
        """
        m, n = X.shape
        X = self.add_theta0(X)
        y_pred = []
        for i in range(m):
            max_index = np.argmax(self.sigmoid(self.all_theta.dot(np.transpose(X[i, :]))))
            y_pred.append(self.list_classes[max_index])

        return y_pred

    def logistic_train_one_class(self, X, y, list_classes, l, c):
        """Train logistic regression for one class

        :param X: train data matrix
        :param y: class labels for training
        :param list_classes: list of categories
        :param l: parameter of regularization lamda
        :param c: index of class to train
        :return: weights used in the logistic function
        """
        m, n = X.shape
        initial_theta = np.zeros(n)
        y_class = self.get_y_class(y, list_classes, c)

        def decorated_cost(theta):
            return self.cost_function_reg(theta, X, y_class, l)

        def decorated_grad(theta):
            return self.grad_function_reg(theta, X, y_class, l)

        theta = fmin_l_bfgs_b(decorated_cost, initial_theta, maxiter=50, fprime=decorated_grad)
        return theta[0]

    def cost_function_reg(self, theta, X, y, l):
        """Calculate value of Cost function

        :param theta: weights
        :param X: data matrix
        :param y: binary vector for particular class
        :param l: parameter of regularization lambda
        :return: cost value
        """
        m, n = X.shape
        J = (1/m) * (-y.T.dot(np.log(self.sigmoid(X.dot(theta)))) - (1-y.T).dot(np.log(1 - self.sigmoid(X.dot(theta)))))+ \
            (l/m)* 0.5 * theta[1:].T.dot(theta[1:])
        return J

    def grad_function_reg(self, theta, X, y, l):
        """Calculate gradient value

        :param theta: weights
        :param X: data matrix
        :param y: binary vector for particular class
        :param l: parameter of regularization lambda
        :return: gradient value
        """
        m, n = X.shape
        grad = (1/m) * X.T.dot(self.sigmoid(X.dot(theta)) - y)
        grad[1:] = grad[1:] + (l/m)*theta[1:]
        return grad

    def sigmoid(self, X):
        """Sigmoid function

        :param X: data matrix
        :return: sigmoid value
        """
        return 1 / (1 + np.exp(-X))

    def add_theta0(self, X):
        """Add free weight (not affected by regularization term) to data matrix

        :param X: data matrix
        :return: data matrix plus free weight
        """
        m, n = X.shape
        X_aux = np.zeros((m, n + 1))
        X_aux[:, 1:] = X
        return X_aux

    def get_y_class(self, y, list_classes, i):
        """Get binary vector from list of classes using one particular class

        :param y: class labels
        :param list_classes: list of classes
        :param i: index of class to use for binarysation
        :return: binary vector
        """
        return np.asarray([b == list_classes[i] for b in y])

