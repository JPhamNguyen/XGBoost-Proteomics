"""(Insert description after this module is done)"""
import pandas as pd
import numpy as np
import sys
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class Datasets:
    """
       Handles data fetching, cleaning, and preprocessing. To assign CSV files to attributes like raw_data or user_data,
       please use the custom CLI arguments to input desired CSV file paths. One example of a CLI command would be:
       'python3 pipeline.py 50 -i Input/<CSV file path>' to make predictions on some other dataset. For more help, use
        the --help flag when inputting CLI commands. The Datasets object will take care of proper assignments and data
        cleaning operations. If you decide to use our dataset (or another) to make predictions on another dataset,
        please make sure your CSV/Excel file matches the format in <Input/database.csv>

       Args: None
       Attributes:
           :pandas DataFrame self._raw_data: holds the raw data from our CSV file
           :pandas DataFrame self._cleaned_data: holds cleaned and one-hot encoded data from our CSV file
           :pandas DataFrame self._user_data: holds the raw data from a user's inputted CSV file
           :pandas DataFrame self._cleaned_user_data: holds cleaned and one-hot encoded data from a user's CSV file
           :pandas Series (?) self._accession_numbers: holds the accession numbers of the proteins in the dataset
           :pandas DataFrame self._x_train: contains the input variables for training
           :pandas DataFrame self._y_train: contains output variable for training
           :pandas Dataframe self._x_test: contains input variables for testing
           :pandas DataFrame self._y_test:contains output variable for testing
           :pandas DataFrame self._target: holds the ground truth values for the output variable
       """
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

    # List of helpful properties to set or get Dataset attributes
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
    def cleaned_data(self):
        if self._cleaned_data is None:
            raise ValueError("Initialize cleaned_data by calling clean_raw_data()")
        else:
            return self._cleaned_data

    @cleaned_data.setter
    def cleaned_data(self, cleaned_raw_data):
        self._cleaned_data = cleaned_raw_data

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

    @property
    def x_train(self):
        if self._x_train is None:
            raise ValueError("Initialize x_train by calling preprocess_data()")
        else:
            return self._x_train

    @x_train.setter
    def x_train(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._x_train = self.read_data(path)
        else:
            # If trying to set to already imported array
            self._x_train = path

    @property
    def y_train(self):
        if self._y_train is None:
            raise ValueError("Initialize x_train by calling preprocess_data()")
        else:
            return self._y_train

    @property
    def x_test(self):
        if self._x_test is None:
            raise ValueError("Initialize x_test by calling preprocess_data()")
        else:
            return self._x_test

    @property
    def y_test(self):
        if self._y_test is None:
            raise ValueError("Initialize y_test by calling preprocess_data()")
        else:
            return self._y_test

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
        Process our dataset, and if specified via the CLI, the user's dataset. Involves cleaning the raw data, setting
        the x_train, y_train, x_test, and y_test datasets, and dropping any unneeded columns from the datasets

        :param: None
        :return: None
        """
        # columns to drop from the dataset(s)
        columns_to_drop = ['Protein Length', 'Sequence', 'Bound Fraction', 'Accession Number']

        # clean our dataset and user's dataset if provided
        self._cleaned_data = clean_raw_data(self.raw_data)

        # Set our training sets from our dataset + grab accession numbers to make it easier to run algorithms like RFECV
        self._y_train = self._target = self._cleaned_data['Bound Fraction'].to_numpy()
        self._accession_numbers = self._cleaned_data['Accession Number']
        self._x_train = self._cleaned_data.drop(labels=columns_to_drop, axis=1)

        # Use our dataset as a training set to make predictions on yours
        if self._user_data is not None:
            self._cleaned_user_data = clean_raw_data(self._user_data)
            self._accession_numbers = self._cleaned_user_data['Accession Number']
            self._y_test = self._cleaned_user_data['Bound Fraction'].to_numpy()
            self._x_test = self._cleaned_user_data.drop(labels=columns_to_drop, axis=1)

        pd.set_option('display.max_columns', None, 'display.max_rows', None)
        np.set_printoptions(threshold=np.inf)

    def split_data(self):
        """
        If the user has not inputted another dataset, split our data into X and Y testing and training sets using
        sklearn's train_test_split (for now at least)

        :param: None
        :return: None
        """
        # attach Accession Numbers to cleaned data in order to maintain proper filtering later one
        self._x_train['Accession Number'] = self._accession_numbers

        # random state set for reproducibility
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self._x_train, self._target,
                                                                                    train_size=0.8, random_state=42)
        # grab shuffled accession numbers
        self._accession_numbers = self._x_test['Accession Number']
        self._x_train = self._x_train.drop('Accession Number', 1)
        self._x_test = self._x_test.drop('Accession Number', 1)

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

    # Max fill for categorical columns + join cleaned DataFrames together (order of features is irrelevant)
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


def scale(dataframe):
    """While tree-based models like XGBoost are relatively insensitive to scaling, feature selection algorithms like
    RFECV are sensitive to large values and thus require scaling to properly work. Works by normalizing and reshaping
    the data by specified columns while preserving labels information.
    Args:
        :param: data (pandas df): The data to normalize
        :param: labels (pandas series): The column labels
    Returns:
        :param data (pandas df): normalized dataframe with preserved column labels
    """
    norm_df = preprocessing.MinMaxScaler().fit_transform(dataframe)
    data = pd.DataFrame(norm_df, columns=list(dataframe))
    data.reset_index(drop=True, inplace=True)
    return data


def rescale(data, labels):
    """Normalize and reshape the data by columns while preserving labels
    information
    Args:
        :param pandas DataFrame data: The data to normalize
        :param pandas Series labels: The column labels
    Returns:
        :param pandas DataFrame data: normalized dataframe with preserved column labels
    """
    norm_df = preprocessing.MinMaxScaler().fit_transform(data)
    data = pd.DataFrame(norm_df,columns=list(data))
    data = pd.concat([labels, data], axis=1)
    data.reset_index(drop=True, inplace=True)
    return data
