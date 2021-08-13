import argparse
import os
import sys
import json
import xgboost as xgb
import optuna
import plotly
import data
import optimize


def pipeline(dataset, iteration, tune_hyper, feature_select):
    # clean dataset(s), set training datasets, and possibly testing datasets if another user's dataset is inputted
    dataset.preprocess_data()

    # tune hyperparameters with Bayesian Optimization via Optuna
    if tune_hyper:
        study = optuna.create_study(direction='minimize', study_name='hyperparameter-tuning')
        study.optimize(lambda trial: optimize.tune_hyperparameters(trial=trial, x_data=dataset.x_train, target=dataset.target), n_trials=100)
        print("\nNumber of finished trials:", len(study.trials))
        print("\nBest Trial:", study.best_trial.params)
        # optuna.visualization.plot_optimization_history(study=study)
        # optuna.visualization.plot_parallel_coordinate(study=study)

        # store hyperparameters as json into a text file to use in the real XGBoost
        if os.path.exists('config/') is False:
            os.mkdir('config')
        with open('config/hyperparameters.txt', 'w') as f:
            json.dump(study.best_params, f)

    # grab previously obtained hyperparameters to use in model
    try:
        with open('config/hyperparameters.txt') as file:
            hyperparameters = json.load(file)
            print(hyperparameters)
    except FileNotFoundError:
        print("\ntxt file containing the hyperparameters is empty, initialize hyperparameters by calling "
              "optimize.tune_hyperparameters()\n")
        sys.exit(1)

    # declare the model (and insert previous hyperparameters --> does it take params as a dictionary?)
    est = xgb.XGBRegressor(**hyperparameters)

    # obtain a binary mask from RFECV
    if feature_select:
        # scale the data features to run the RFECV algorithm
        dataset.x_train = data.scale(dataset.x_train)
        optimize.feature_selection(model=est, x_train=dataset.x_train, y_train=dataset.y_train, output_file='Input/_mask.txt')

    sys.exit(0)

    # go through ML pipeline
    for i in iteration:
        print(f'Run Number: {i}')
        run_and_evaluate(cleaned_dataset=dataset, xgb=est)


def run_and_evaluate(cleaned_dataset, xgb):
    """A custom pipeline for running, testing, and evaluating the XGBoost model """

    # split our dataset to make predictions
    if cleaned_dataset.user_data is None:
        cleaned_dataset.split_data()

    # apply RFECV mask to extract optimal features
    # apply the binary mask obtained with RFECV
    data.x_train, data.x_test = data.apply_RFECV_mask('Input/_mask.txt', data.x_train, data.x_test)
    print(data.x_train.columns)

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
    parser.add_argument('-f', '--feature', help="Set to true to perform feature selection via RFECV")

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
    pipeline(dataset=dataset, iteration=iterations, tune_hyper=tune_hyperparameters, feature_select=feature_selection)

