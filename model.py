import argparse
import os
import sys
import data


class XGBoostModel:
    def __init__(self):
        self.model = None
        self.hyper_parameters = {}
        self.base_models = []


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
    parser.add_argument('-o', '--optimize', help="set 'True' to optimize hyper-parameters",
                        type=bool, default=False)

    # parse CLI commands
    args = vars(parser.parse_args(sys.argv))
    iterations = args['iterations']
    optimize = args['optimize']

    # Set our dataset
    dataset = data.dataset()
    dataset.raw_data = args['train']

    # Set user's dataset if provided
    if args['input'] is not None:
        dataset.user_data = args['input']

    # preprocess and clean the dataset(s)
    dataset.process_data()

    # go through ML pipeline


