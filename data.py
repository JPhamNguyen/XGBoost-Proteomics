""" This module holds our XGBoostModel object, and the data preprocessing and cleaning is handled here as well """
import pandas as pd
import numpy as np
import sys
import os

import sklearn.impute
from sklearn.impute import SimpleImputer


class dataset:
    def __init__(self):
        self._raw_data = None
        self._cleaned_data = None
        self._user_data = None
        self._cleaned_user_data = None
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None

    # List of helpful properties to set or get XGBoostModel attributes
    @property
    def raw_data(self):
        if self._raw_data is None:
            raise ValueError("Initialize raw data by setting raw_data=<path.csv>")
        else:
            return self._raw_data

    @raw_data.setter
    def raw_data(self, enm_database):
        if isinstance(enm_database, str) and os.path.isfile(enm_database):
            self._raw_data = self.read_data(enm_database)
        else:
            raise ValueError("Please provide existing filepath to desired dataset")

    @property
    def user_data(self):
        if self._user_data is None:
            raise ValueError("Initialize user data by setting the --i tag in CLI arguments")
        else:
            return self._user_data

    @user_data.setter
    def user_data(self, enm_database):
        if isinstance(enm_database, str) and os.path.isfile(enm_database):
            self._user_data = self.read_data(enm_database)
        else:
            raise ValueError("Please provide existing filepath to desired dataset")

    @property
    def cleaned_user_data(self):
        if self._cleaned_user_data is None:
            raise ValueError("Initialize cleaned_user_data by calling clean_raw_data()")
        else:
            return self._cleaned_user_data

    @cleaned_user_data.setter
    def cleaned_user_data(self, cleaned_raw_data):
        self._cleaned_user_data = cleaned_raw_data

    @property
    def cleaned_data(self):
        if self._cleaned_data is None:
            raise ValueError("Initialize cleaned_data by calling clean_raw_data()")
        else:
            return self._cleaned_data

    @cleaned_data.setter
    def cleaned_data(self, cleaned_raw_data):
        self._cleaned_data = cleaned_raw_data

    @staticmethod
    def read_data(enm_database):
        """Read a CSV filepath and return the CSV file as a pandas Dataframe

        :param dataset object self: the dataset object currently being acted upon
        :param str enm_database: a filepath to an existing desired CSV file
        :return pandas Dataframe enm_database: returns the CSV file as a pandas DataFrame
        """
        # The filepath will have already been checked before returning the CSV file as a DataFrame
        return pd.read_csv(enm_database)

    def process_data(self):
        """
        Process our dataset--and if specified via the CLI--and the user's dataset to make predictions on

        :param: None
        :return: None
        """
        # clean our dataset and user's dataset if provided
        self._cleaned_data = clean_raw_data(self.raw_data)

        if self._user_data is not None:
            self._cleaned_user_data = clean_raw_data(self._user_data)
        
        # create the X-train
        if self.user_data is not None:
            self._cleaned_user_data = clean_raw_data(self.user_data)

    def split_data(self):
        """
        If the user has not inputted another dataset, split our data into X and Y testing and training sets using
        (insert method)

        :param: None
        :return: None
        """


def clean_raw_data(dataset):
    """
    Clean and preprocess the dataset. Perform any imputations on missing values, and also one-hot encode
    categorical variables for the XGBoostRegressor

    :param Dataframe dataset: the raw dataset to be cleaned and preprocessed
    :return Dataframe dataset: the cleaned and preprocessed dataset
    """
    # categorical variables to one-hot encode (add additional categorical variables to this list in the
    # future to preprocess and clean
    categorical_data = ['Enzyme Commission Number', 'Particle Size', 'Particle Charge',
                        'Solvent Cysteine Concentration', 'Solvent NaCl Concentration']
    columns_to_drop = ['Protein Length', 'Sequence', 'Accession Number', 'Bound Fraction']
    nonnumerical = ['Accession Number', 'Sequence']

    # separate non-numerical + categorical from continuous variables in order to perform imputations if necessary
    nonnumerical_and_categorical = categorical_data + nonnumerical
    nonnumerical_data = dataset[nonnumerical]
    categorical_data = dataset[categorical_data]
    continuous = dataset.drop(labels=nonnumerical_and_categorical, axis=1)

    # Fill missing continuous values with corresponding average value
    imputer = SimpleImputer(strategy="mean")
    imputed_continuous = imputer.fit_transform(continuous)
    continuous = pd.DataFrame(imputed_continuous, columns=continuous.columns, index=continuous.index)

    # Max fill function for categorical columns
    for category in categorical_data.columns:
        categorical_data[category].fillna(categorical_data[category].value_counts().idxmax(), inplace=True)

    # cleaned_dataset = continuous.join(categorical_data, axis=1).join(nonnumerical_data, axis=1)
    # Bug: get a SettingWithCopyWarning when trying to join dataframes together --> probably because of chained calls
    sys.exit(0)
    return cleaned_dataset