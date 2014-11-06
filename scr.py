import pandas as pd
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

y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
#print("Number of mislabeled points out of a total %d points : %d"
#      % (iris.data.shape[0],(iris.target != y_pred).sum()))

clat_target["income"] = 1
gnbf = mnb.fit(clat_data.values, clat_target.values)




import math
import functools

class GaussianNaiveBayes(object):
    def __init__(self):
        #self.prev_targets = []
        self.prev_targets = [iris.target[0], iris.target[1], iris.target[50], iris.target[100]]
        #self.prev_unique_targets = []
        self.prev_unique_targets = sorted(list(set(self.prev_targets)))

        #self.prev_variances = []
        #1st_feature_var = var(iris.data[0][0], iris.data[1][0], iris.data[2][0]...)
        #self.prev_variances =

        self.prev_means = []
        # prev row for prediction
        #self.prev_fp_row = None





        self.features = []#{}
        self.feature_count = Counter()
        # How many samples already loaded into object
        self.sample_count = 0

    '''
    def probability_of_given_evidence(self, target):
        prob_sex = self.feature_count[" Male"]

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
        self.prev_unique_targets = sorted(list(set(self.prev_targets)))

        self.sample_count += 1

        for feature_values, value in zip(self.features, row):
            feature_values[value] += 1

        print self.features
        probability_target = self.probability_of_given_evidence(target)
   '''

    def train_row(self, row, target):
        self.prev_targets.append(target)
        self.prev_unique_targets = sorted(list(set(self.prev_targets)))

        #for i in row:
        #    prev_variances.append()
        #self.prev_variances.append(
        #1st_feature_var = var(iris.data[0][0], iris.data[1][0], iris.data[2][0]...)
        #self.prev_variances =

        self.prev_means = []

        pass

    def calculate_gaussian_evidence_likelyhood(self, value, m):
        feature_variance = self.prev_variances[m]
        feature_mean = self.prev_means[m]

        # prev_variances[m] takes the variance of the right feature
        k = 1 / math.sqrt(2 * math.pi * feature_variance)
        ex = (-1 * (value - feature_mean)**2) / 2 * feature_variance
        return k * math.exp(ex)

    # non_normalized_posterior
    # Actually this is posterior*normalizing constant
    def calculate_posterior(self, row, target, n):
        #P(income 1)
        prior = self.prev_targets.count(target)/float(self.prev_targets)
        evidence_likelyhoods = []
        for m, value in enumerate(row):
            gel = calculate_gaussian_evidence_likelyhood(value, m)
            evidence_likelyhoods.append(gel)
        #evidence_likelyhood =

        # We don't need this
        #normalizing_constant =
        #
        #P(age|income 1)
        mult = lambda a, b: a * b
        return prior * functools.reduce(mult, evidence_likelyhoods)


    def predict(self, row):
        posteriors = []
        for n, target in enumerate(self.prev_targets):
            posteriors.append(calculate_posterior(row, target, n))
        return max(posteriors)
        # self.current_posterios : {0:1.23, 1:2.5, ...}
        #return biggest_value(current_posteriors)








