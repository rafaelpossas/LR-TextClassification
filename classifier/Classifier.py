import numpy as np
import pandas as pd
import sklearn.cross_validation as cross_validation



class MultinomialNaiveBayes():

    def __init__(self):
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.classes = []

    def score(self,result,expected):
        size = len(result)
        count = 0
        for index in range(size):
            result_class = result[index]
            expected_class = expected[index]
            if result_class == expected_class:
                count += 1
        return count/size

    def predict(self,y):
        n_docs,n_words = y.shape
        class_prob = {}
        result = []
        for doc in range(n_docs):
            indices = np.where(y[doc] > 0)[0]
            for clazz in self.classes:
                class_prior = self.prior[clazz]
                class_prob[clazz] = class_prior
                joint_prob = self.likelihood[clazz][indices]
                class_prob[clazz] += joint_prob.sum()

            selected_class = max(class_prob.keys(), key=lambda k: class_prob[k])
            result.append(selected_class)
        return result

    def fit(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        self.classes = np.unique(y)
        
        ###########################
        # Calculating the Prior Probabilities for the classes
        # pos_count and neg_count gives the count for each word for a particular class
        class_word_count = {}
        word_count = {}
        class_count = {}
        class_prior = {}

        # examining each word and fining the above mentioned values
        cwLength = x.shape[1]
        cdLength = x.shape[0]

        for doc in range(n_docs):
            clazz = y[doc][0]
            if clazz in class_count:
                class_count[clazz] += 1
            else:
                class_count[clazz] = 1

        for c_w in range(cwLength):
            word_count[c_w] = 0
            for c_d in range(cdLength):
                clazz = y[c_d][0]
                value = x[c_d][c_w]
                #Finding the value of that word for a particular class
                if clazz in class_word_count :
                    class_word_count[clazz][c_w] += value
                else:
                    class_word_count[clazz] = np.zeros(n_words)
                    class_word_count[clazz][c_w] += value

                word_count[c_w] += value # Counting how many times that word appears in all documents

        # Finding the Prior Probability for all classes

        # Finding likelihood for each word with respective to a class
        for c_w in range(cwLength):
            for clazz in self.classes:
                numerator = class_word_count[clazz][c_w]
                denominator = class_word_count[clazz].sum()
                joint = (numerator + 1) / (denominator + n_words)
                class_prior[clazz] = np.log(class_count[clazz] / n_docs)
                class_word_count[clazz][c_w] = np.log(joint)

        ###########################
        self.likelihood = class_word_count
        self.prior = class_prior
        self.trained = True

if __name__ == '__main__':
    X = pd.read_csv('../assignment1_2016S1/training_data.csv', header=None).sort_values(by=0).reset_index().ix[:100, 2:]
    y = pd.read_csv('../assignment1_2016S1/training_labels.csv', header=None).sort_values(by=0).reset_index().ix[:100, 2:]

    nb = MultinomialNaiveBayes()

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)

    nb.fit(X_train.as_matrix(), y_train.as_matrix())

    result = nb.predict(X_test.as_matrix())


    print("Accuracy: {0}%".format(nb.score(result,y_test.as_matrix())*100))
