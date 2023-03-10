import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


class KNN:

    def __init__(self, k=5):

        self.k = k

    def fit(self, input_data, target):

        self.X = input_data
        self.Y = target

    def set_new_k(self, k):
        self.k = k

    def metric(self, x1, x2):

        return np.sum((x1 - x2) ** 2)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, input_data):

        distances = [self.metric(input_data, i) for i in self.X]

        minn = np.argsort(distances)[:self.k]
        labels = [self.Y[i] for i in minn]

        mst_common = Counter(labels).most_common(1)

        return mst_common[0][0]

    #     validation

    #     def validate(self,split_ = 5):

    # #         self.performace_over_validation = {}
    #         mean_accuracy = []
    #         list_k = []

    #         accuracy = 0
    #         best_k = 1

    #         for k in range(1,int(np.sqrt(self.X.shape[0]))):

    #             X_train,X_test,Y_train,Y_test = train_test_split(self.X,self.Y,test_size=0.2)

    #             accuracies = []

    #             for i in range(split_):

    #                 model = KNN(k)
    #                 model.fit(X_train,Y_train)

    #                 prediction = model.predict(X_test)

    #                 _accuracy = model.accuracy(prediction,Y_test)
    #                 accuracies.append(_accuracy)

    #             mean_accuracy.append(np.mean(accuracies))
    #             list_k.append(k)

    #             if np.mean(accuracies) > accuracy:
    #                 best_k = k

    #         return mean_accuracy,list_k,best_k

    def validate(self):

        k_list = []
        mean_accuraccies = []

        for k in range(1, 30):

            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2)

            accuraccies = []

            for j in range(10):
                model = KNN(k)
                model.fit(X_train, Y_train)
                #                 print(model)
                #                 print(X_test)
                predicted = model.predict(X_test)
                accuraccies.append(self.accuracy(predicted, Y_test))

            k_list.append(k)
            mean_accuraccies.append(np.mean(accuraccies))

        return k_list, mean_accuraccies

    #     accuracy metrics

    def accuracy(self, prediction, real):
        return round(100 * (prediction == real).sum() / prediction.shape[0])

    def precision(self, prediction, real):
        TruePositive = np.where((prediction == 1) & (real == 1), 1, 0).sum()
        FalsePositive = np.where((prediction == 1) & (real == 0), 1, 0).sum()

        prec = round(100 * TruePositive / (TruePositive + FalsePositive))

        return prec

    def sensitivity(self, prediction, real):

        TruePositive = np.where((prediction == 1) & (real == 1), 1, 0).sum()
        FalseNegative = np.where((prediction == 0) & (real == 1), 1, 0).sum()

        rec = round(100 * TruePositive / (TruePositive + FalseNegative))

        return rec

    def specificity(self, prediction, real):

        TrueNegative = np.where((prediction == 0) & (real == 0), 1, 0).sum()
        FalseNegative = np.where((prediction == 1) & (real == 0), 1, 0).sum()

        trn = round(100 * TrueNegative / (TrueNegative + FalseNegative))

        return trn

    def FalseNegativeRate(self, prediction, real):

        FalseNegative = np.where((prediction == 0) & (real == 1), 1, 0).sum()
        TruePositive = np.where((prediction == 1) & (real == 1), 1, 0).sum()

        fnr = round(100 * FalseNegative / (FalseNegative + TruePositive))
        return fnr

    def FalsePositiveRate(self, prediction, real):

        FalsePositive = np.where((prediction == 1) & (real == 0), 1, 0).sum()
        TrueNegative = np.where((prediction == 0) & (real == 0), 1, 0).sum()

        fpr = round(100 * FalsePositive / (FalsePositive + TrueNegative))

        return fpr

    def f1_score(self, prediction, real):

        TruePositive = np.where((prediction == 1) & (real == 1), 1, 0).sum()
        FalseNegative = np.where((prediction == 0) & (real == 1), 1, 0).sum()
        FalsePositive = np.where((prediction == 1) & (real == 0), 1, 0).sum()

        rec = TruePositive / (TruePositive + FalseNegative)
        prec = TruePositive / (TruePositive + FalsePositive)

        f1 = round(100 * (2 * prec * rec) / (prec + rec))

        return f1

    def class_performace_measures(self, predicition, real):

        #         TruePositive = np.where((prediction == 1) & (real == 1),1,0).sum()
        #         FalsePositive = np.where((prediction == 1) & (real == 0),1,0).sum()

        #         TrueNegative = np.where((prediction == 0) & (real == 0),1,0).sum()
        #         FalseNegative = np.where((prediction == 0) & (real == 1),1,0).sum()

        #         acc = round(100 * (TruePositive + TrueNegative)/(TruePositive + TrueNegative + FalsePositive + FalseNegative))

        performance_measures = {"accuracy": self.accuracy(prediction, real),
                                "precision": self.precision(prediction, real),
                                "sensitivity": self.sensitivity(prediction, real),
                                "specificity": self.specificity(prediction, real),
                                "false_negative_rate": self.FalseNegativeRate(prediction, real),
                                "false_positive_rate": self.FalsePositiveRate(prediction, real),
                                "f1_score": self.f1_score(prediction, real)}

        return performance_measures
