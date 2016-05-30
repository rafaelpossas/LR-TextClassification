import time, logging
from classifier.assignment import load_data, get_precision_recall_fscore_overall, cross_validation
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import xgboost as xgb

def main():
    logging.basicConfig(filename='resultsXGB.log', filemode='w', level=logging.INFO)
    logging.info('Started')
    ## load data

    start_time = time.time()
    logging.info('Loading data')
    X_train, y_train, X_test = load_data()
    list_classes = list(set(y_train))
    list_classes.sort()
    y_train = label_to_index(y_train,list_classes)
    m, n = X_train.shape
    k = 3
    logging.info("--- %s seconds ---" % (time.time() - start_time))

    models = []
    eta = [0.7]
    num_round = [120]
    colsample_bytree = [0.5]
    param = [(i, j, k) for i in eta for j in num_round for k in colsample_bytree]

    for p in param:
        start_time = time.time()
        logging.info('XGBoost with {} rounds, {} of eta, '
                     'and colsample_bytree {}'.format(p[1], p[0], p[2]))
        print('XGBoost with {} rounds, {} of eta, '
                     'and colsample_bytree {}'.format(p[1], p[0], p[2]))
        results = []
        c_cross = 0
        for training, validation in cross_validation(k, m):
            print('cross validation iteration {}'.format(c_cross))
            y_train_cross = [y_train[y] for y in training]
            y_val_cross = [y_train[y] for y in validation]

            xg_train = xgb.DMatrix(X_train[training], label=y_train_cross)
            xg_test = xgb.DMatrix(X_train[validation], label=y_val_cross)

            # setup parameters for xgboost
            param = {}
            # use softmax multi-class classification
            param['objective'] = 'multi:softmax'
            # scale weight of positive examples
            param['eta'] = p[0]
            param['max_depth'] = 6
            param['colsample_bytree'] = p[2]
            param['silent'] = 1
            param['nthread'] = 8
            param['num_class'] = 30

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]
            num_round = p[1]
            bst = xgb.train(param, xg_train, num_round, watchlist)
            # get prediction
            y_pred = bst.predict(xg_test)

            res = precision_recall_fscore_support(y_val_cross, y_pred, average='micro')
            results.append(res)
            c_cross += 1

        results = get_precision_recall_fscore_overall(results, k)

        logging.info(results)
        logging.info("--- %s seconds ---" % (time.time() - start_time))

    logging.info('Finished')

def label_to_index(labels, list_classes):
    return np.array([list_classes.index(i) for i in labels], dtype='f')


if __name__ == "__main__":
    main()