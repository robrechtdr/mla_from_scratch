import math
import functools
from collections import OrderedDict
import numpy as np


# The implementation was mainly derived from the explanation of the
# Gaussian based Naive Bayes algorithm in the following source:
# http://en.wikipedia.org/wiki/Naive_Bayes_classifier
class GaussianNaiveBayes(object):
    """A Gaussian distribution based implementation of the Naive Bayes
    algorithm.
    """
    def __init__(self):
        self.targets = OrderedDict()
        self.variances = []
        self.means = []

    @property
    def unique_targets(self):
        return self.targets.keys()

    def get_summary_statistic_index(self, target):
        """Get the index for a summary statistic (i.e. mean or variance)
        corresponding to a target value.

        Args:
            target (any value): A target value.

        Returns:
            int.

        """
        return self.unique_targets.index(target)

    # An incremental form instead of the classical algorithm is used. See
    # 'incremental algorithm' in
    # http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance .
    #
    # Such an incremental form makes it possible to have a much more
    # memory efficient version of the Gaussian Bayes algorithm by merely storing
    # the means and variances for each feature per target. Without an
    # incremental form we would need to store each feature value of each
    # record per target!
    #
    # If the data consisted of say 10,000 records with 4 features and 10 types
    # of occuring target values, then the incremental form only needs to store
    # 80 (=2 * 4 * 10) float values as opposed to the non-incremental form
    # which would need to store 40,000 (=4 * 10,000) values.
    # In this calculation, the target dictionary is not counted but would in
    # both cases have the same size and be small (containing 10 key-value pairs).
    def _mean(self, prev_mean, n, value):
        """Calculate the mean.

        Args:
            prev_mean (float): The mean of n - 1  values.

            n (int): The total number of values.

            value (int or float): The nth value.

        Returns:
            float.

        """
        return ((n - 1) * prev_mean + value) / float(n)

    # See comments about _mean.
    def _variance(self, prev_variance, n, value, curr_mean, prev_mean):
        """Calculate the variance.

        Args:
            prev_variance (float): The variance of n - 1 values.

            n (int): The total number of values.

            value (int or float): The nth value.

            curr_mean (float): The mean of n values.

            prev_mean (float): The mean of n - 1 values.

        Returns:
            float.

        """
        k = (n - 1) * prev_variance
        numerator = k + (value - prev_mean) * (value - curr_mean)
        return numerator / float(n)

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

            # Update summary statistics.
            self.means[ss_ind][m] = curr_mean
            self.variances[ss_ind][m] = curr_variance

    def fit(self, data, targets):
        """Train the algorithm by associating rows of feature values with
        respective target values.

        Args:
            data (numpy.array): A two dimensional array where each nested
                array represents a record where each feature value is a
                number.

            targets (numpy.array): A one dimensional array where each item
                represents a target value associated with a respective
                record.

        Returns:
            None.

        """
        if len(data) != len(targets):
            raise ValueError("Shape of data and targets don't correspond")
        for row, target in zip(data, targets):
            self._fit_row(row, target)

    def _calc_gaussian_evidence_likelyhood(self, value, n, m):
        feature_variance = self.variances[n][m]
        feature_mean = self.means[n][m]

        # The exponent calculation can not only result in a ZeroDivisionError
        # but also in a nan (E.g. 0.0/0.0).
        ex_denom = float(2 * feature_variance)
        if ex_denom == 0:
            # A negative infinite exponent in the gaussian distribution
            # function has a 0 as result.
            return 0
        else:
            ex = (-1 * (value - feature_mean)**2) / ex_denom
            k = 1 / float(math.sqrt(2 * math.pi * feature_variance))
            return k * math.exp(ex)

    # Since for all different target values the normalizing constant has the
    # the same value, ignoring it in the calculation will not affect the
    # selection of the target as the one with the highest probability.
    #
    # So while getting the exact absolute target probabilities is not necessary,
    # it is necessary to get the correct proportions of the target
    # probabilities relative to one another.
    def _calc_non_norm_posterior(self, row, target, n):
        """Calculate the non-normalized posterior.

        With 'non-normalized posterior' is meant the following:
        posterior * normalizing constant.

        Args:
            row (numpy.array): A one dimensional array of numerical values.

            target (any value): A target value.

        Returns:
            float.
        """
        t_count = self.targets[target]
        prior = t_count / float(sum(self.targets.values()))
        evidence_likelyhoods = []
        for m, value in enumerate(row):
            gel = self._calc_gaussian_evidence_likelyhood(value, n, m)
            # We implement the following because in the end all evidence
            # likelyhoods are multiplied and the algorithm will completely
            # fail to predict if there is one zero valued evidence likelyhood
            # or more per target.
            # So a negligibly small value is used instead of 0 if it occurs.
            if gel == 0:
                gel = 10**-10
            evidence_likelyhoods.append(gel)

        mult = lambda a, b: a * b
        return prior * functools.reduce(mult, evidence_likelyhoods)

    def _predict_row(self, row):
        # Non-normalized posteriors.
        nn_posteriors = {}
        for n, target in enumerate(self.unique_targets):
            nn_posteriors[target] = self._calc_non_norm_posterior(row, target, n)

        result = max(zip(nn_posteriors.values(), nn_posteriors.keys()))
        return result[1]

    def predict(self, data):
        """Predict the target values of an array of records.

        Args:
            data (numpy.array): A two dimensional array where each nested
                array represents a record.

        Returns:
            numpy.array.

        """
        if not self.targets:
            raise Exception("Can't predict without any training")

        result = []
        for row in data:
            result.append(self._predict_row(row))
        return np.array(result)
