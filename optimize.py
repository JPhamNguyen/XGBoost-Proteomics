"""This module contains tools to either optimize the XGBoost hyper-parameters or perform feature selection """
import argparse
import sys
import pandas as pd
import numpy as np
import data
import pipeline
import os
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


def tune_hyperparameters(trial, x_data, target):
    """Use Bayesian Optimization for comprehensive and time-efficient hyperparameter tuning, and save those hyperparameters
    as JSON into a text file for later use. This is a black box function that optimizes hyperparameters based on the
    RMSE score of the XGBoostRegressor.
    NOTE: current hyperparameters can be overwritten when running back-to-back hyperparameter tunings

    :param: None
    :return: None
    """
    # Set testing boundaries for desired hyperparameters
    # To see list of hyperparameters, use this link: https://xgboost.readthedocs.io/en/latest/parameter.html
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2500),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'gamma': trial.suggest_float('gamma', .01, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 6),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.2, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 4.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.0, 1.0)
    }

    # Accession Numbers are only used for identification and filtering of proteins
    # check if x_data is being passed in by reference or value
    x_data = x_data.drop('Accession Number', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x_data, target, train_size=0.8, random_state=42)

    # run the model, calculate the RSME score, and use the RSME score as the objective function to minimize in future
    # trials in the Bayesian Optimization process
    model = xgb.XGBRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=10, verbose=True)
    preds = model.predict(x_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse


def feature_selection():
    pass


def test_baseline_models():
    print("Performing baseline model testing")


if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('b', 'base', help="Set to True to test baseline models against XGBoost", default=False)
    parser.add_argument('-tr', '--train', help="file path to your csv file train model on", type=pipeline.check_file_path,
                        default='Input/database.csv')
    parser.add_argument('-t', '--tune', help="set 'True' to tune hyper-parameters", type=bool, default=False)
    parser.add_argument('f', '--feature', help="Set to true to perform feature selection via RFECV ")

    # parse CLI commands
    args = vars(parser.parse_args(sys.argv))

    # Set our dataset
    dataset = data.Datasets()
    dataset.raw_data = args['train']
