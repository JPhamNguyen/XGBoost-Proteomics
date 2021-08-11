"""This module contains tools to either optimize the XGBoost hyper-parameters or perform feature selection """
import argparse
import sys
import data


def random_search_cv():
    print("Performing RandomSearchCV")


def feature_selection():
    print("Performing feature selection")


def tune_hyperparameters():
    print("Manually tuning hyper-parameters")


def test_baseline_models():
    print("Performing baseline model testing")


if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('b', 'base', help="Set to True to test baseline models against XGBoost", default=False)
    parser.add_argument('-tr', '--train', help="file path to your csv file train model on", type=check_file_path,
                        default='Input/database.csv')
    parser.add_argument('-t', '--tune', help="set 'True' to tune and test hyper-parameters", type=bool, default=False)
    parser.add_argument('f', '--feature', help="Set to true to perform feature selection via RFECV ")

    # parse CLI commands
    args = vars(parser.parse_args(sys.argv))

    # Set our dataset
    dataset = data.Datasets()
    dataset.raw_data = args['train']

    #