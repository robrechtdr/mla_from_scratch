import math
import functools
from collections import OrderedDict
import numpy as np


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


