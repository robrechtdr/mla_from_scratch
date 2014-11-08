import pandas as pd
import numpy as np
import sklearn
'''
# See adult.names
columns = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "captital-gain", "capital-loss", "hours-per-week", "native-country",
           "income"]
train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                    # Needs to be here explicitly. File doesn't have any header.
                    header=None,
                    names=columns)

test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                   # Needs to be here explicitly, replaces current 'would be
                   # header' with columns.
                   header=0,
                   names=columns)


# First, check for each columns all possible used values. This is to detect
# corrupted rows.

def print_value_oversight(df):
    # Amount of columns is small so we can do this. (what is pandas syntax?)
    for column in df.columns:
        print column
        print df[column].unique()

# This reveals there are no NA values (there are ? values though)
print_value_oversight(train)


def clean_df(df):
    # there are no NA columns
    # First let's get rid of all NA values
    #
    # This necessary?
    # Then get rid of the '?' in the workclass, occupation and narive-country
    # column
    return df

ctrain = clean_df(train)


# Perhaps better to change <=50k to 1 and other to 0. (no, only do when
# necess.)
target = ctrain["income"]
ctrain.drop("income", axis=1)

# Now let's train our classifier.


# Now let's test on the test data



clat = train.head()[['sex', 'workclass', 'income']]
clat_data = clat[["sex", "workclass"]]
clat_target = clat["income"]

clat_data.to_csv("clat_data.csv")
clat_target.to_csv("clat_target.csv")
'''

clat_data = pd.read_csv("clat_data.csv")
del clat_data["Unnamed: 0"]
clat_target = pd.read_csv("clat_target.csv", header=None)
clat_target["income"] = clat_target[1]
del clat_target[0]
del clat_target[1]

from collections import Counter


# Start with implementation for data with categorical features only

# A non-naive bayes classifier would have to be able to
# evaluate wether variables would be independent between each
# variable. => False, experimentally is always dependent.
# The problem would be calculating P(B and C ...|A)?
class NaiveBayes(object):
    def __init__(self):
        self.features = []#{}
        self.feature_count = Counter()
        # How many samples already loaded into object
        self.sample_count = 0

    def probability_of_given_evidence(self, target):
        prob_sex = self.feature_count[" Male"]

        #P("income"|"sex and workclass") = P("")
        #P(A)
        #P(B)
        #P(A|B)
        #P(B|A)

        #return probability
        pass

    def on_initial_train(self, row):
        # Should only need to happen on first training
        # Make unnames list of features
        for n, i in enumerate(row):
            self.features.append(Counter())

    def train_row(self, row, target):
        if not self.features:
            self.on_initial_train(row)
            #for n, i in enumerate(row):
            #    self.features.append(Counter())
        ##########

        self.sample_count += 1

        for feature_values, value in zip(self.features, row):
            feature_values[value] += 1

        print self.features
        probability_target = self.probability_of_given_evidence(target)


nb = NaiveBayes()
# Use numpy arrays as input.
cd = clat_data.values[0]
ct = clat_target.values[0][0]
nb.train_row(cd, ct)


from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
gnb = GaussianNB()
mnb = MultinomialNB()

#y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
#print("Number of mislabeled points out of a total %d points : %d"
#      % (iris.data.shape[0],(iris.target != y_pred).sum()))

clat_target["income"] = 1
#gnbf = mnb.fit(clat_data.values, clat_target.values)










import math
import functools
from collections import OrderedDict


class GaussianNaiveBayes(object):
    def __init__(self):
        self.targets = OrderedDict()
        self.variances = []
        self.means = []

    @property
    def unique_targets(self):
        return self.targets.keys()

    # Get summary statistic(e.g. mean or variance) index
    def get_summary_statistic_index(self, target):
        return self.unique_targets.index(target)

    # Another implementation would be to use the sum of the values and count
    # of a given feature
    def _mean(self, prev_mean, t_count, value):
        return ((t_count - 1) * prev_mean + value) / float(t_count)

    # See incremental algorithm:
    # http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # To calculate variance from previous variance and nr of values for a given
    # feature.
    # Saves us from having to store all values of a given feature!!!
    # This is an example of the benefit of being able to derive a recursive
    # definition of a mathematical relation.
    def _variance(self, prev_variance, t_count, value, curr_mean, prev_mean):
        k = (t_count - 1) * prev_variance
        numerator = k + (value - prev_mean) * (value - curr_mean)
        return numerator / float(t_count)

    def _fit_row(self, row, target):
        # Only on first occurence of a new target value
        if target not in self.unique_targets:
            ind = len(self.targets)
            self.means.append([])
            self.variances.append([])
            for value in row:
                # Mean of only one value is the value
                self.means[ind].append(value)
                # Variance of only one element is 0
                self.variances[ind].append(0)

            self.targets[target] = 1

        # Only on subsequent occurence of a target value
        else:
            self.targets[target] += 1

        ss_ind = self.get_summary_statistic_index(target)
        t_count = self.targets[target]

        for m, value in enumerate(row):
            curr_mean = self._mean(self.means[ss_ind][m], t_count, value)
            curr_variance = self._variance(self.variances[ss_ind][m],
                                     t_count, value, curr_mean,
                                     self.means[ss_ind][m])
            # update old values
            self.means[ss_ind][m] = curr_mean
            self.variances[ss_ind][m] = curr_variance


    def fit(self, data, targets):
        if len(data) != len(targets):
            raise ValueError("data and target don't correspond")
        for row, target in zip(data, targets):
            self._fit_row(row, target)

    def _calc_gaussian_evidence_likelyhood(self, value, n, m):
        # If not floatified, will return nan instead of zerodiv error.
        feature_variance = self.variances[n][m]
        feature_mean = self.means[n][m]

        ex_denom = float(2 * feature_variance)
        # A negative infinite exponent in the gaussian function has
        # a 0 as result. To prevent a ZeroDivisionError, an try except
        # form is not implemented because sometimes nan is returned instead
        # of a ZeroDivisionError being raised.
        if ex_denom == 0:
            return 0
        else:
            ex = (-1 * (value - feature_mean)**2) / ex_denom
            k = 1 / float(math.sqrt(2 * math.pi * feature_variance))
            return k * math.exp(ex)


        # A negative infinite exponent in the gaussian function has
        # a 0 as result
        #except ZeroDivisionError:
        #    return 0
        #else:
        #    k = 1 / float(math.sqrt(2 * math.pi * feature_variance))
        #    return k * math.exp(ex)

    # With non normalized posterior, posterior*normalizing constant is meant.
    def _calc_non_norm_posterior(self, row, target, n):
        t_count = self.targets[target]
        prior = t_count / float(sum(self.targets.values()))
        evidence_likelyhoods = []
        for m, value in enumerate(row):
            gel = self._calc_gaussian_evidence_likelyhood(value, n, m)
            # A correction on the naive bayes alg to avoid incapability
            # of prediction in case of one or more zero values evidence
            # likelyhoods per target
            if gel == 0:
                # Because in the end all evidence likelyhoods are multiplied,
                # the algorithm will completely fail to predict
                # if there is one zero valued variance or more per target.
                # So use a negligibly small value instead.
                gel = 10**-10
            evidence_likelyhoods.append(gel)

        mult = lambda a, b: a * b
        return prior * functools.reduce(mult, evidence_likelyhoods)

    def _predict_row(self, row):
        nn_posteriors = {}
        for n, target in enumerate(self.unique_targets):
            nn_posteriors[target] = self._calc_non_norm_posterior(row, target, n)

        result = max(zip(nn_posteriors.values(), nn_posteriors.keys()))
        print nn_posteriors
        return result[1]

    def predict(self, data):
        if not self.targets:
            raise Exception("Can't predict without any training")

        result = []
        for row in data:
            result.append(self._predict_row(row))
        return np.array(result)


from sklearn import cross_validation
data_train, data_test, target_train, target_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

gnb = GaussianNaiveBayes()
gnb.fit(data_train, target_train)

sk_gnb = sklearn.naive_bayes.GaussianNB()
sk_gnb.fit(data_train, target_train)

gnb_pred = gnb.predict(data_test)
sk_gnb_pred = sk_gnb.predict(data_test)


def get_prediction_stats(predicted_targets, real_targets):
    assert len(predicted_targets) == len(real_targets)
    amount_of_targets = len(real_targets)
    incorrectly_predicted = []
    for predicted, real in zip(predicted_targets, real_targets):
        if predicted != real:
            incorrectly_predicted.append({"real": real, "predicted": predicted})

    amount_incorrect = len(incorrectly_predicted)
    amount_correct = amount_of_targets - amount_incorrect
    accuracy = 1 - amount_incorrect/float(amount_of_targets)

    print ("{0} out of {1} targets were predicted correctly "
           "({2}% correct).".format(
        amount_correct, amount_of_targets, accuracy * 100))
    return accuracy, incorrectly_predicted, amount_of_targets

gp = get_prediction_stats(gnb_pred, target_test)
sk_gp = get_prediction_stats(sk_gnb_pred, target_test)

