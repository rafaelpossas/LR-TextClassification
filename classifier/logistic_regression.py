import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time, logging
from classifier.assignment import load_data, cross_validation, get_precision_recall_fscore_overall
from sklearn.metrics import precision_recall_fscore_support
from joblib import Parallel, delayed
import multiprocessing

def main():

    logging.basicConfig(filename='resultslogistic.log', filemode='w', level=logging.INFO)
    logging.info('Started')

    start_time = time.time()
    logging.info('Loading data')
    X_train, y_train, X_test = load_data()
    m, n = X_train.shape
    list_classes = list(set(y_train))
    list_classes.sort()
    logging.info("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    logging.info('Reduce matrix')
    columns = (X_train != 0).sum(0)
    X_train = X_train[:,columns<500]
    m, n = X_train.shape
    logging.info('X_train shape {},{}'.format(m,n))
    logging.info("--- %s seconds ---" % (time.time() - start_time))


    k = 10
    ls = [0.3]

    for l in ls:
        results = []
        c_cross = 0
        logging.info('Logistic Regression with lambda {}'.format(l))
        print('Logistic Regression with lambda {}'.format(l))
        start_time = time.time()
        for training, validation in cross_validation(k, m):
            print('cross validation iteration {}'.format(c_cross))
            y_train_cross = [y_train[y] for y in training]
            y_val_cross = [y_train[y] for y in validation]
            all_theta = logistic_train(X_train[training], y_train_cross, list_classes, l)
            y_pred = logistic_test(X_train[validation], all_theta, list_classes)
            res = precision_recall_fscore_support(y_val_cross, y_pred, average='micro')
            results.append(res)
            c_cross += 1

        logging.info(get_precision_recall_fscore_overall(results, k))
        logging.info("--- %s seconds ---" % (time.time() - start_time))

def logistic_train(X, y, list_classes, l):
    classes = len(list_classes)
    X = add_theta0(X)
    num_cores = multiprocessing.cpu_count() -1
    results = Parallel(n_jobs=num_cores)(delayed(logistic_train_one_class)(X, y, list_classes, l,c) for c in range(classes))
    return np.asarray(results)

def logistic_train_one_class(X, y, list_classes,l ,c):
    m, n = X.shape
    initial_theta = np.zeros(n)
    y_class = get_y_class(y, list_classes, c)

    def decorated_cost(theta):
        return cost_function_reg(theta, X, y_class, l)

    def decorated_grad(theta):
        return grad_function_reg(theta, X, y_class, l)

    theta = fmin_l_bfgs_b(decorated_cost, initial_theta, maxiter=50, fprime=decorated_grad)
    return theta[0]

def logistic_test(X, all_theta, list_classes):
    m, n = X.shape
    X = add_theta0(X)
    y_pred = []
    for i in range(m):
        max_index = np.argmax(sigmoid(all_theta.dot(np.transpose(X[i,:]))))
        y_pred.append(list_classes[max_index])

    return y_pred

def cost_function_reg(theta, X, y, l):
    m, n = X.shape
    J = (1/m) * (-y.T.dot(np.log(sigmoid(X.dot(theta)))) - (1-y.T).dot(np.log(1 - sigmoid(X.dot(theta))))) + \
        (l/m)* 0.5 * theta[1:].T.dot(theta[1:])
    return J

def grad_function_reg(theta, X, y, l):
    m, n = X.shape
    grad = (1/m) * X.T.dot(sigmoid(X.dot(theta)) - y)
    grad[1:] = grad[1:] + (l/m)*theta[1:]
    return grad

def predict(X, theta):
    y = sigmoid(X.dot(theta))
    return y>0.5

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def add_theta0(X):
    m, n = X.shape
    X_aux = np.zeros((m, n + 1))
    X_aux[:, 1:] = X
    return X_aux

def get_y_class(y, list_classes, i):
    return np.asarray([b == list_classes[i] for b in y])

if __name__ == "__main__":
    main()
