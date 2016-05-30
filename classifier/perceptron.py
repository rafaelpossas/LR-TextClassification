import numpy as np
import math

def perceptron_train(X_train, y_train, iterations, list_classes):
    m, n = X_train.shape
    classes = len(list_classes)
    w = np.zeros((classes, n), dtype='f')
    v = np.zeros((classes, n), dtype='f')
    for i in range(iterations):
        print('starting iteration {}'.format(i))
        for j in range(m):

            y_p = get_winning_class(X_train[j], w, list_classes, classes)

            if not y_train[j] == y_p:
                index_train = get_index(y_train[j],list_classes)
                index_pred = get_index(y_p, list_classes)
                w[index_train] += X_train[j]
                w[index_pred] -= X_train[j]
            v += w

    return v/(iterations*m)

def perceptron_test(X_test, w_avg, list_classes):

    classes = len(list_classes)
    m = len(X_test)
    y_pred = list()

    for i in range(m):

        y_p = get_winning_class(X_test[i], w_avg, list_classes, classes)

        y_pred.append(y_p)
    return y_pred

def get_class(c, classes):
    cat = np.zeros(classes)
    cat[c] = 1
    return cat

def get_index(c, list_classes):
    return list_classes.index(c)

def get_winning_class(x_train, w, list_classes, classes):
    arg_max, y_p = -math.inf, 0
    for c in range(classes):
        current_activation = np.dot(x_train, w[c])
        if current_activation >= arg_max:
            arg_max, y_p = current_activation, list_classes[c]
    return y_p
