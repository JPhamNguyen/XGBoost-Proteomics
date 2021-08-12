"""This module contains tools to either optimize the XGBoost hyper-parameters or perform feature selection """
import argparse
import sys
import data
import pipeline
import os
import optuna
from sklearn.feature_selection import RFECV


def feature_selection(model, x_train, y_train, output_file):
    """Select most optimal features by using RFECV (recursive feature elimination cross validation.
    Save the output as a binary mask to apply to the cleaned dataset to extract optimal features later.

    :param: None
    :returns: None
    """
    assert isinstance(output_file, str), "please pass a string specifying mask-file location"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mask_file = os.path.join(dir_path, output_file)

    selector = RFECV(estimator=model, step=1, cv=5, scoring='neg_mean_squared_error', verbose=1)
    selector = selector.fit(x_train, y_train)

    # display optimal features
    feature_index = selector.get_support(indices=True)
    features = []

    for index in feature_index:
        features.append(x_train.columns[index])

    print("selector support: \n {} \n selector ranking: \n {}".format(selector.support_, selector.ranking_))
    print("Optimal number of features: \n {} \n Selector grid scores: \n {} \n".format(selector.n_features_, selector.grid_scores_))
    print("Features Indexes: \n{}\n".format(feature_index))
    print("Feature Names: \n{}".format(features))

    # write optimum binary mask to text file
    with open(mask_file, 'w') as f:
        for item in selector.support_:
            f.write('{}, '.format(item))


def tune_hyperparameters(trial, data, target):
    """Use Bayesian Optimization for comprehensive and time-efficient hyperparameter tuning, and save those hyperparameters
    as JSON into a text file for later use.
    NOTE: current hyperparameters can be overwritten when running back-to-back hyperparameter tunings

    :param: None
    :return: None
    """
    # Set testing boundaries for desired hyperparameters
    # To see list of hyperparameters, use this link: https://xgboost.readthedocs.io/en/latest/parameter.html
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2500),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'gamma': trial.suggest_float('gamma', .01, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 6),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.2, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 4.0, 10.0),
        'reg_labmda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.0, 1.0)
    }

    # run the model, calculate the RSME score, and use the RSME score as the objective function to minimize in future
    # trials in the Bayesian Optimization process


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
