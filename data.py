"""(Insert description after this module is done)"""
import pandas as pd
import sys
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class Datasets:
    def __init__(self):
        self._raw_data = None
        self._cleaned_data = None
        self._user_data = None
        self._cleaned_user_data = None
        self._accession_numbers = None
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self._target = None

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

    @property
    def accession_numbers(self):
        if self._accession_numbers is None:
            raise ValueError("Initialize accession_numbers by calling clean_raw_data()")
        else:
            return self._accession_numbers

    @property
    def target(self):
        if self._target is None:
            raise ValueError("Initialize accession_numbers by calling clean_raw_data()")
        else:
            return self._target

    @staticmethod
    def read_data(enm_database):
        """Read a CSV filepath and return the CSV file as a pandas Dataframe

        :param dataset object self: the dataset object currently being acted upon
        :param str enm_database: a filepath to an existing desired CSV file
        :return pandas Dataframe enm_database: returns the CSV file as a pandas DataFrame
        """
        # The filepath will have already been checked before returning the CSV file as a DataFrame
        return pd.read_csv(enm_database)

    def preprocess_data(self):
        """
        Process our dataset, and if specified via the CLI, the user's dataset

        :param: None
        :return: None
        """
        # columns to drop from the dataset(s)
        columns_to_drop = ['Protein Length', 'Sequence', 'Bound Fraction']

        # clean our dataset and user's dataset if provided
        self._cleaned_data = clean_raw_data(self.raw_data)

        # Set our training sets from our dataset
        self._y_train = self._target = self._cleaned_data['Bound Fraction']
        self._x_train = self._cleaned_data.drop(labels=columns_to_drop, axis=1)

        # Use our dataset as a training set to make predictions on yours
        if self._user_data is not None:
            self._cleaned_user_data = clean_raw_data(self._user_data)
            self._accession_numbers = self._cleaned_user_data['Accession Number']
            self._y_test = self._cleaned_user_data['Bound Fraction']

            # drop some unneeded columns
            self._x_test = self._cleaned_user_data.drop(labels=columns_to_drop, axis=1)
            self._x_test = self._x_test.drop('Accession Number', axis=1)
            self._x_train = self._x_train.drop('Accession Number', axis=1)
            print(self._x_test.columns)
        else:
            # split our dataset to make predictions
            self.split_data()
        # nothing to return, simply proceed with the pipeline

    def split_data(self):
        """
        If the user has not inputted another dataset, split our data into X and Y testing and training sets using
        (insert method)

        :param: None
        :return: None
        """
        # random state set for reproducibility
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self._x_train, self._target,
                                                                                    train_size=.8, random_state=42)
        self._accession_numbers = self._x_train['Accession Number']
        print(self._x_train.shape, self._x_test.shape, self._y_train.shape, self._y_test.shape)
        print(self._accession_numbers)
        sys.exit(0)


def clean_raw_data(raw_data):
    """
    Clean and preprocess the dataset. Perform any imputations on missing values, and also one-hot encode
    categorical variables

    :param Dataframe raw_data: the raw dataset to be cleaned and preprocessed
    :return Dataframe dataset: the cleaned and preprocessed dataset
    """
    # Check that there are no missing Accession Numbers
    assert raw_data['Accession Number'].isna().any() is not False, \
        "Please check and replace missing Accession Numbers"

    # categorical variables to one-hot encode (add additional categorical variables to this list in the
    # future to preprocess and clean)
    categorical = ['Enzyme Commission Number', 'Particle Size', 'Particle Charge',
                   'Solvent Cysteine Concentration', 'Solvent NaCl Concentration']
    nonnumerical = ['Accession Number', 'Sequence']

    # separate non-numerical + categorical from continuous variables and imputate
    nonnumerical_and_categorical = categorical + nonnumerical
    nonnumerical_data = raw_data[nonnumerical]
    categorical_data = raw_data[categorical]
    continuous = raw_data.drop(labels=nonnumerical_and_categorical, axis=1)

    # Fill missing continuous values with corresponding average value
    imputer = SimpleImputer(strategy="mean")
    imputed_continuous = imputer.fit_transform(continuous)
    continuous = pd.DataFrame(imputed_continuous, columns=continuous.columns, index=continuous.index)

    # Max fill function for categorical columns + join cleaned DataFrames together (order of features is irrelevant)
    categorical_data = categorical_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
    cleaned_dataset = continuous.join(categorical_data).join(nonnumerical_data)

    # one-hot encode categorical columns
    for category in categorical:
        cleaned_dataset = one_hot_encode(cleaned_dataset, category)
    return cleaned_dataset


def one_hot_encode(dataframe, category):
    """This function converts categorical variables into one hot vectors
    Args:
        :param Pandas DataFrame dataframe: Dataframe containing column to be encoded
        :param str category: specifying the column to encode
    Returns:
        :return Pandas DataFrame dataframe: With the specified column now encoded into a one
        hot representation
    """
    assert isinstance(dataframe, pd.DataFrame), 'data argument needs to be pandas dataframe'
    dummy = pd.get_dummies(dataframe[category], prefix=category)
    dataframe = pd.concat([dataframe, dummy], axis=1)
    dataframe.drop(category, axis=1, inplace=True)
    return dataframe

