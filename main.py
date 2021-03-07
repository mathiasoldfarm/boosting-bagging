from base import base, base_type, base_params
from read import X_train, X_test, y_train, y_test
from bagging import BaggingCounting
from adaboost import AdaboostM1
from param_search import ParamSearch
import pandas as pd
from log import log, reset_log

reset_log()
base.fit(X_train, y_train)
base_train_score = base.score(X_train, y_train)
base_test_score = base.score(X_test, y_test)

log(f"Base train score: {base_train_score}")
log(f"Base test score: {base_test_score}")
log("")

bagging = BaggingCounting(base=base_type, n_bags=200, max_depth=1, max_leaf_nodes=2)
bagging.fit(X_train, y_train)
bagging_train_score = bagging.score(X_train, y_train)
bagging_test_score = bagging.score(X_test, y_test)

log(f"Bagging train score: {bagging_train_score}")
log(f"Bagging test score: {bagging_test_score}")
log("")

params = {
  "base": [base_type],
  "n_bags": [100, 200, 500, 1000],
  "max_depth": [4, 12, 24, 64],
  "max_leaf_nodes": [4, 24, 64, 128]
}

paramsearch = ParamSearch(BaggingCounting, params)
paramsearch.search(X_train, y_train, 20)
paramsearch.find_best()

paramsearch.fit(X_train, y_train)

param_search_test_score = paramsearch.score(X_test, y_test)
log(f"Bagging with best params, test score: {param_search_test_score}")

stump = base_type(**{"max_depth": 1, "max_leaf_nodes": 2})
stump.fit(X_train, y_train)
stump_train_score = stump.score(X_train, y_train)
stump_test_score = stump.score(X_test, y_test)

log(f"STUMP train score: {stump_train_score}")
log(f"STUMP test score: {stump_test_score}")
log("")

adaboost = AdaboostM1(base=base_type, iterations=200, max_depth=1, max_leaf_nodes=2)
adaboost.fit(X_train, y_train)
adaboost_train_score = adaboost.score(X_train, y_train)
adaboost_test_score = adaboost.score(X_test, y_test)

log(f"Adaboost train score: {adaboost_train_score}")
log(f"Adaboost test score: {adaboost_test_score}")

params = {
  "base": [base_type],
  "iterations": [100, 200, 500, 1000],
  "max_depth": [1, 2, 4, 8],
  "max_leaf_nodes": [2, 8, 16, 32]
}

paramsearch = ParamSearch(AdaboostM1, params)
paramsearch.search(X_train, y_train, 20)
paramsearch.find_best()

paramsearch.fit(X_train, y_train)

param_search_test_score = paramsearch.score(X_test, y_test)
log(f"Adaboost with best params, test score: {param_search_test_score}")