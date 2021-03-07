import numpy as np
from log import log
import pandas as pd

class ParamSearch:
  def __init__(self, learner, params):
    self.learner = learner
    self.params = params
    self.history = []
    self.best_params = None
    self.final_learner = None

  def search(self, X_train, y_train, iterations):
    log("Searching for params...")
    log("")
    self.history = []
    self.best_params = None

    for _ in range(iterations):
      selected_params = {}
      for key in self.params:
        selected_value = self.params[key]
        selected_params[key] = np.random.choice(selected_value, 1)[0]

      splits = np.array_split(X_train.join(y_train, sort=False), 5)
      validation_scores = []
      for i in range(5):
        train = splits.copy()
        validation = splits[i]
        del train[i]
        train = pd.concat(train, sort=False)

        y_cross = train["label"]
        X_cross = train.drop(columns="label")
        y_cross_val = validation["label"]
        X_cross_val = validation.drop(columns="label")

        learner = self.learner(**selected_params)
        learner.fit(X_cross, y_cross)
        validation_score = learner.score(X_cross_val, y_cross_val)
        validation_scores.append(validation_score)

      final_validation_score = sum(validation_scores) / 5
      self.history.append((selected_params, final_validation_score))

      log(f"Params: {selected_params}")
      log(f"5-fold Validation score: {final_validation_score}")
      log("")

  def find_best(self):
    best_history = sorted(self.history, key=lambda x: x[1], reverse=True)[0]
    log(f"Best params: {best_history[0]}")
    log(f"Best 5-fold validation score: {best_history[1]}")
    log("")

    self.best_params = best_history[0]

  def fit(self, X, y):
    self.final_learner = self.learner(**self.best_params)
    self.final_learner.fit(X, y)

  def score(self, X, y):
    return self.final_learner.score(X, y)

  


