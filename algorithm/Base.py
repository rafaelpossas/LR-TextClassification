import csv, random, statistics as stat
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

class Base:

    def __init__(self, filename, targetname, test_name):
        self.dataset_name = filename
        self.test_name = test_name
        self.target_name = targetname

    def load_data(self):
        """load data from files

        :return: data training matrix X_train, training labels y_train, data testing matrix X_test,
        and names used for testing
        """

        X_train = self.extract_data(self.dataset_name)
        y_train = self.extract_data(self.target_name, array = False)
        X_test, test_names = self.extract_data(self.test_name, sort = False, names = True)

        return X_train, y_train, X_test, test_names

    def dimension_reduction(self, X_train, option, value):
        """dimensionsionality reduction application

        :param X_train: data matrix
        :param option: reduction process (common: remove columns with more than 'value' rows with values > 0
        (considered as stopwrods), pca: apply pca to data matrix, commonpca: apply both process together)
        :param value: value to use in process
        :return: processed data matrix
        """
        if 'common' in option:
            if not value:
                value = 500
            columns = (X_train != 0).sum(0)
            X_train = X_train[:, columns < value]
        if 'pca' in option:
            if not value:
                value = 0.95
            pca = PCA(n_components=value)
            X_train = pca.fit_transform(X_train)
        return X_train

    def extract_data(self,filename, sort= True, array = True, names = False):
        """Extract data from files

        :param filename: path to file to extract
        :param sort: if it is true, sort the data according to first column (names of mobile applications)
        :param array: if it is true, return a list of second columns value (use for extract labels)
        :param names: if it is true, extract data matrix and first colummn (use for testing)
        :return: return data according parameters
        """
        content = []
        with open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                content.append(row)

            if sort: content.sort(key=lambda x: x[0])

            if array:
                content_array = [x[1:] for x in content]
                content_array = np.asarray(content_array, dtype='f')
            else:
                content_array = [y[1] for y in content]

            if names:
                names_list = [n[0] for n in content]
                return content_array, names_list

        return content_array


    def save_data(self,content, filename):
        """Save content in path 'filename'

        :param content: content to save
        :param filename: path to save a file
        :return:
        """
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for row in content:
                writer.writerow(row)

    def cross_validation(self,k, m):
        """Yield cross validation indexes

        :param k: numbers of folds to use
        :param m: number of instances
        :return: training and validation indexes
        """
        items = list(range(m))
        random.shuffle(items)
        slices = [items[i::k] for i in range(k)]
        for i in range(k):
            validation = slices[i]
            training = [item
                        for s in slices if s is not validation
                        for item in s]
            yield training, validation

    def get_precision_recall_fscore_overall(self,results):
        """Get overall results for a list of results

        :param results: list of results
        :return: average and standard deviation of metrics
        """
        precision, recall, fscore = [], [], []
        for res in results:
            precision.append(res[0])
            recall.append(res[1])
            fscore.append(res[2])
        return stat.mean(precision), stat.stdev(precision), stat.mean(recall), \
               stat.stdev(recall), stat.mean(fscore), stat.stdev(fscore)

    def confusion_matrix(self, y_test, y_pred):
        list_classes = sorted(list(set(y_test)))
        cm = np.zeros([len(list_classes),len(list_classes)], dtype=int)
        for i in range(len(y_test)):
            cm[list_classes.index(y_test[i]),list_classes.index(y_pred[i])] += 1
        return cm

    def get_precision_recall_fscore(self, y_test, y_pred):
        precision, recall, fscore = [], [], []
        cm = self.confusion_matrix(y_test, y_pred)
        ## macro averaging
        for i in range(cm.shape[0]):
            t_p = cm[i,i]
            f_p = np.sum(cm[:, i])-cm[i,i]
            f_n = np.sum(cm[i, :])-cm[i,i]
            pre = t_p/(t_p+f_p)
            rec = t_p/(t_p+f_n)
            fs = 2*pre*rec/(pre+rec)
            precision.append(pre)
            recall.append(rec)
            fscore.append(fs)
        return stat.mean(precision), stat.mean(recall), stat.mean(fscore)

    def plot_confusion_matrix(self, y_test, y_pred, list_classes):
        cm = self.confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(list_classes))
        plt.xticks(tick_marks, list_classes, rotation=90)
        plt.yticks(tick_marks, list_classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.grid(True)
        width, height = len(list_classes), len(list_classes)
        for x in range(width):
            for y in range(height):
                if cm[x][y] > 100:
                    color = 'white'
                else:
                    color = 'black'
                if cm[x][y] > 0:
                    plt.annotate(str(cm[x][y]), xy=(y, x),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 color=color)
        plt.show()

    def plot_histogram(self, y_train):
        ## Make histogram of category distribution
        counter = Counter(y_train).most_common()
        classes = [c[0] for c in counter]
        classes_f = [c[1] for c in counter]

        # Plot histogram using matplotlib bar().
        indexes = np.arange(len(classes)) + 0.5
        width = 0.5
        rec = plt.bar(indexes, classes_f, width)
        plt.xticks(indexes + width * 0.5, classes, fontsize=15, rotation=90)
        plt.ylabel('Frequency', fontsize=16)
        plt.xlabel('Classes', fontsize=16)
        plt.title('Frequency Histogram for Classes', fontsize=16)
        plt.tight_layout()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2., height + 1,
                         '%d' % int(height),
                         ha='center', va='bottom', fontsize=16)

        autolabel(rec)
        plt.show()