import argparse
import os
import sys
import xgboost as xgb
import data
import optimize
import json


def pipeline(data, iteration, tune_hyper, feature_select):
    # clean + set x_training, x_test, y_training, and y_testing datasets
    data.preprocess_data()

    # optimize XGBoost hyperparameters and save it as JSON
    if tune_hyper:
        optimize.tune_hyperparameters(output_file='hyperparameters.txt')

    try:
        with open('hyperparameters.txt') as json_file:
            hyperparameters = json.load(json_file)
            print(hyperparameters.type())
            print(hyperparameters)
    except ValueError:
        print("txt file containing the hyperparameters is empty, initialize hyperparameters by calling optimize.tune_hyperparameters")
        hyperparameters = {}

    # declare the model (and insert previous hyperparameters --> does it take params as a dictionary?)
    est = xgb.XGBRegressor()

    # obtain a binary mask from RFECV
    if feature_select:
        optimize.feature_selection(model=est, x_train=data.x_train, y_train=data.y_train, output_file='Input/_mask.txt')

        # go through ML pipeline
    for i in iteration:
        print(f"Run Number: {i}")
        run_and_evaluate(cleaned_dataset=dataset, xgb=est)


def run_and_evaluate(cleaned_dataset, xgb):
    """A custom pipeline for running, testing, and evaluating the XGBoost model """

    # split our dataset to make predictions
    if cleaned_dataset.user_data is None:
        cleaned_dataset.split_data()

    # apply RFECV mask to extract optimal features

    # fit, run, and iterate on previous versions of the XGBoost


def check_file_path(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path
    else:
        print("\nInvalid file path. Please input an existing filepath to desired file.\n")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', help="script with which to run the model", type=str)
    parser.add_argument('iterations', help="number of times to run the model", type=int)
    parser.add_argument('-t', '--train', help="file path to your csv file train model on", type=check_file_path,
                        default='Input/database.csv')
    parser.add_argument('-i', '--input', help="file path to your csv file to make predictions on", type=check_file_path,
                        default=None)
    parser.add_argument('-tu', '--tune', help="set 'True' to tune and test hyper-parameters", type=bool, default=False)
    parser.add_argument('f', '--feature', help="Set to true to perform feature selection via RFECV")

    # parse CLI commands
    args = vars(parser.parse_args(sys.argv))
    tune_hyperparameters = args['tune']
    feature_selection = args['feature']
    iterations = args['iterations']

    # Set our dataset
    dataset = data.Datasets()
    dataset.raw_data = args['train']

    # Set user's dataset if provided
    if args['input'] is not None:
        dataset.user_data = args['input']

    # run the pipeline to clean data, tune hyperparameters, perform feature selection, and run the model
    pipeline(data=dataset, iteration=iterations, tune_hyper=tune_hyperparameters, feature_select=feature_selection)

