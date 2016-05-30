from classifier.perceptron import perceptron_test, perceptron_train
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from classifier.assignment import load_data, cross_validation, precision_recall_fscore_support
from classifier.assignment import get_precision_recall_fscore_overall, make_binary
import logging, time
import itertools
import operator

def main():

    logging.basicConfig(filename='resultscombined.log', filemode='w', level=logging.INFO)
    logging.info('Started')
    ## load data

    start_time = time.time()
    logging.info('Loading data')
    X_train, y_train, X_test = load_data()
    list_classes = list(set(y_train))
    list_classes.sort()
    m, n = X_train.shape
    k = 3
    logging.info("--- %s seconds ---" % (time.time() - start_time))

    models = [MultinomialNB(), LogisticRegression()]

    start_time = time.time()
    X_train = make_binary(X_train, 0)
    logging.info('Combined algorithm')
    print('Combined algorithm')
    results = get_results_algorithms(X_train, y_train, m, k, models, list_classes)
    logging.info(results)
    logging.info("--- %s seconds ---" % (time.time() - start_time))

def get_results_algorithms(X_train, y_train, m, k, models, list_classes):
    results = []
    c_cross = 0
    iterations = 3
    for training, validation in cross_validation(k, m):
        print('cross validation iteration {}'.format(c_cross))

        y_train_cross = [y_train[y] for y in training]
        y_val_cross = [y_train[y] for y in validation]

        models[0].fit(X_train[training], y_train_cross)
        y_pred0 = models[0].predict(X_train[validation])

        models[1].fit(X_train[training], y_train_cross)
        y_pred1 = models[1].predict(X_train[validation])

        w_avg = perceptron_train(X_train[training], y_train_cross, iterations, list_classes)
        y_pred2 = perceptron_test(X_train[validation], w_avg, list_classes)

        y_pred = [most_common((y_pred0[i],y_pred1[i],y_pred2[i])) for i in range(len(y_pred0))]

        res = precision_recall_fscore_support(y_val_cross, y_pred, average='micro')
        results.append(res)
        c_cross += 1

    return get_precision_recall_fscore_overall(results, k)


def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

if __name__ == "__main__":
    main()