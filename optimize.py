"""This module contains tools to either optimize the XGBoost hyper-parameters or perform feature selection """
import argparse
import sys
import data
import pipeline
import os
from bayes_opt import BayesianOptimization
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

    # view RFECV accuracy scores
    # visualization_utils.visualize_rfecv(selector.grid_scores_)

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


def tune_hyperparameters(output_file):
    """Use Bayesian Optimization for comprehensive and time-efficient hyperparameter tuning, and save those hyperparameters
    as JSON into a text file for later use.
    NOTE: current hyperparameters can be overwritten when running back-to-back hyperparameter tunings

    :param: None
    :return: None
    """
    # Set search space boundaries, initial test points, and other parameters
    bayesian_optimization_maximize_params = dict(
        init_points=10,  # init_points=20,
        n_iter=30,  # n_iter=60,
        acq='poi', xi=0.0
    )
    bayesian_optimization_boundaries = dict(
        n_estimators=(100, 3000),
        max_depth=(5, 15),
        gamma=(0.01, 5),
        min_child_weight=(0, 6),
        scale_pos_weight=(1.2, 5),
        reg_alpha=(4.0, 10.0),
        reg_lambda=(1.0, 10.0),
        max_delta_step=(0, 5),
        subsample=(0.5, 1.0),
        colsample_bytree=(0.3, 1.0),
        learning_rate=(0.0, 1.0)
    )
    bayesian_optimization_initial_search_points = dict(
        n_estimators=[100, 2500],
        max_depth=[5, 10],
        gamma=[0.1511, 3.8463],
        min_child_weight=[2.4073, 4.9954],
        scale_pos_weight=[2.2281, 4.0345],
        reg_alpha=[2.501, 9.0573],
        reg_lambda=[2.0126, 3.5934],
        max_delta_step=[1, 3],
        subsample=[0.650, 0.8234],
        colsample_bytree=[0.395, 0.7903],
        learning_rate=[0.05, 0.7]
    )


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
