import argparse
import os
import sys
import xgboost as xgb
import data
import optimize


def pipeline(tune_hyper, feature_select):
    """A custom pipeline for running, testing, and evaluating the XGBoost model """
    # optimize the hyperparameters here
    # pull hyperparameters from a .txt file?
    # can also update hyperparameters and write those into a .txt file?
    hyperparameters = {}

    model = xgb.XGBRegressor()


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
    iterations = args['iterations'], tune = args['tune'], feature = args['feature']

    # Set our dataset
    dataset = data.Datasets()
    dataset.raw_data = args['train']

    # Set user's dataset if provided
    if args['input'] is not None:
        dataset.user_data = args['input']

    # clean + preprocess dataset(s)
    dataset.preprocess_data()

    # go through ML pipeline
    for i in iterations:
        pipeline(tune_hyper=tune, feature_select=feature)


