from sklearn import tree
import pandas as pd
from random import randrange

class BaggingCounting:
  def __init__(self, base, n_bags=200, max_depth=1, max_leaf_nodes=2):
    self.base = base
    self.n_bags=n_bags
    self.max_depth=max_depth
    self.max_leaf_nodes=max_leaf_nodes

  def sample(self, X, y, sample_ratio=0.2):
    original_size = X.shape[0]
    sample_size = int(original_size * sample_ratio)

    X_sampled, y_sampled = [], []
    for _ in range(sample_size):
      sample_index = randrange(original_size)
      X_sampled.append(X.iloc[sample_index])
      y_sampled.append(y.iloc[sample_index])

    return X_sampled, y_sampled


  def fit(self, X_train, y_train):
    learners = []
    for _ in range(self.n_bags):
      learner = self.base(max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes)
      X_sampled, y_sampled = self.sample(X_train, y_train)
      learner.fit(X_sampled, y_sampled)
      learners.append(learner)
    self.learners = learners

  def predict(self, X):
    bags = []
    for learner in self.learners:
      learner_prediction = learner.predict(X)
      bags.append(learner_prediction)

    bags = pd.DataFrame(bags)
    predictions = bags.apply(lambda column: column.mode())

    return predictions.values[0]

  def score(self, X, y):
    predictions = self.predict(X)
    
    corrects = 0
    for prediction, label in zip(predictions, y):
      if prediction == label:
        corrects += 1

    return corrects / len(X)
