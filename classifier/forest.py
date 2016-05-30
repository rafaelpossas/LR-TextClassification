from sklearn.ensemble import RandomForestClassifier
import logging, time
from classifier.assignment import load_data, get_results_algorithms

def main():
    logging.basicConfig(filename='resultsforest.log', filemode='w', level=logging.INFO)
    logging.info('Started')
    ## load data

    start_time = time.time()
    logging.info('Loading data')
    X_train, y_train, X_test = load_data()
    list_classes = list(set(y_train))
    list_classes.sort()
    m, n = X_train.shape
    k = 10
    logging.info("--- %s seconds ---" % (time.time() - start_time))

    models = []
    trees = [10, 50, 75, 100, 125, 150, 200]

    for n in trees:
        models.append(RandomForestClassifier(n_estimators=n))

    for model in models:
        start_time = time.time()
        logging.info('RandomForestClassifier')
        results = get_results_algorithms(X_train, y_train, m, k, model)
        logging.info(results)
        logging.info("--- %s seconds ---" % (time.time() - start_time))

    logging.info('Finished')

if __name__ == "__main__":
    main()