import numpy as np
import random

class AdaboostM1:
  def __init__(self, base, iterations, max_depth, max_leaf_nodes):
    self.base = base
    self.learners = []
    self.significances = []
    self.iterations = iterations
    self.max_depth = max_depth
    self.max_leaf_nodes = max_leaf_nodes

  def fit(self, X, y):
    data_size = X.shape[0]
    weights = np.array([1 / data_size] * data_size)

    self.learners = []
    self.significances = []

    for _ in range(self.iterations):
      learner = self.base(max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes)
      learner.fit(X, y, sample_weight=weights)
      predictions = learner.predict(X)
      error = (np.where(y != predictions, 1, 0) * weights).sum() / weights.sum()
      significance = (1/2)*np.log((1-error) / error)
      weights *= np.exp(-y * predictions * significance)

      self.learners.append(learner)
      self.significances.append(significance)

  def predict(self, X):
    predictions = [learner.predict(X) for learner in self.learners]
    predictions = np.sign(np.array(self.significances)@np.array(predictions))
    return predictions

  def score(self, X, y):
    predictions = self.predict(X)
    corrects = np.sum(np.where(predictions == y, 1, 0))
    return corrects / len(X)
