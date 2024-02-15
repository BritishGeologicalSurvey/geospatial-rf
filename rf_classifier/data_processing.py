# -*- coding: utf-8 -*-
"""
Various functions to support pre and post random forest 
model application data processing 

@author: ahall
"""
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from skimage.measure import label
import random
import fnmatch
import logging


def get_rows_with_nans(dataframe):
    """Returns dataframe 

    Variables
        dataframe - input dataframe to check for nans

    Get rows containing nans to inspect (maybe you need to 
    sort one of the variables)
    """
    return dataframe[dataframe.isnull().any(axis=1)]


def drop_invalid_results(dataframe, rock_presence_column):
    """Returns dataframe 

    Drops all rows where the rock presence is not 1 or 0

    Variables:
        dataframe - pandas dataframe 
        rock_presence_column - rock presence value column name

    """
    values = [1, 0]
    new_df = dataframe[dataframe[rock_presence_column].isin(values)]

    drops_total = len(dataframe) - len(new_df)

    print("dropped " + str(drops_total) + " rows due to invalid rock presence values")
    return new_df


def categorize_aspect(pandas_series, check_output=False):
    """Returns a numpy.ndarray 

    Recategorizes numerical aspect values to quadrants (N/NE/E etc.)

    North:     0° – 22.5°
    Northeast: 22.5° – 67.5°
    East:      67.5° – 112.5°
    Southeast: 112.5° – 157.5°
    South:     157.5° – 202.5°
    Southwest: 202.5° – 247.5°
    West:      247.5° – 292.5°
    Northwest: 292.5° – 337.5°
    North:     337.5° – 360°
    
    Variables:
        pandas_series - pandas data series of aspect values (0-360°) (expected to be 1D)
            e.g. categorize_aspect(dataframe['your_column'])
        check_output - boolean variable in case user wants to do an ad hoc sense check of recategorization
    """

    bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5,
            247.5, 292.5, 337.5, 360]
    names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N2']  # have to be unique

    aspect_cat = pd.cut(pandas_series, bins, labels=names)
    aspect_cat[aspect_cat == 'N2'] = 'N'  # merge N2 cats with N

    # Check values have mapped correctly
    if check_output:
        rndom_indxs = np.random.randint(low=1, high=aspect_cat.shape[0],
                                        size=20)  # random values within shape of series
        before = pandas_series[rndom_indxs]
        after = aspect_cat[rndom_indxs]
        print("Aspect (deg N) | Aspect quadrant")
        for i in range(0, len(before)):
            print("%0.2f  | %s" % (before.iloc[i], after.iloc[i]))

    return aspect_cat


def min_max_scale(pandas_series, verbose=True):
    """Returns: numpy.ndarray  

    Takes in a pandas data frame series (expected to be 1D)
    Scales values between 0 and 1
    e.g. min_max_scale(dataframe['your_column'])   

    Variables:
        pandas_series - input pamndas.Series object 
        verbose - boolean to control command line print out
    """

    min_max_scaler = preprocessing.MinMaxScaler()

    if verbose:
        print("******************")
        print(pandas_series.name, "\n")
        print("Before rescaling:")
        print("Min: ", min(pandas_series.values))
        print("Max: ", max(pandas_series.values))

    minmax = min_max_scaler.fit_transform(pandas_series.values.reshape(-1, 1))  # reshaped as needs a 2D array

    if verbose:
        print("\nAfter rescaling:")
        print("Min: ", min(minmax))
        print("Max: ", max(minmax))
        print("******************")
    return minmax


def rescale_data(input_df, rock_presence_column, x_col='x', y_col='y', z_col='z'):
    """ Returns a pandas.DataFrame

    Rescale numeric columns of dataframe only of type int* or float*

    Variables:
        input_df - input pandas.DataFrame
        rock_presence_column - (string) name of dependent variable column
        x_col - name of x coordinate column
        y_col - name of y coordinate column
        z_col - name of z coordinate column (variable column)
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  # there must be a better way than this
    numeric_cols = input_df.select_dtypes(include=numerics)

    features_rescaled = input_df.copy()
    assert id(features_rescaled) != id(input_df)  # check the copy is different

    for x in input_df.columns:
        if x not in [x_col, y_col, z_col, rock_presence_column]:
            if x in numeric_cols:
                features_rescaled[x] = min_max_scale(input_df[x])
            else:
                features_rescaled[x] = input_df[x].astype('category')  # non numerics become categoricals

    return features_rescaled


def label_contiguous_points(features, interval, rock_presence_column, x, y):
    """Returns pandas.DataFrame 

    Applies a unique label to every contiguous set of points

    Variables:
        features - (DataFrame) dataframe of training data
        interval -  (int) spatial resolution of the data in metres
        rock_presence_column - (string) name of dependent variable column
        x -  (string) name of the x coordinate column
        y -  (string) name of the y coordinate column

    """

    min_x = min(features[x])
    min_y = min(features[y])

    features['id'] = range(1, len(features) + 1)
    features['x_coord'] = (features.loc[:, x] - min_x) / interval
    features['y_coord'] = (features.loc[:, y] - min_y) / interval

    # Convert x_coord and y_coord to ints, or later coordinate and feature merge won't work and neither
    # will indexing using x_coord and y_coord.
    features = features.astype({"x_coord": int, "y_coord": int})

    max_x_coord = int(max(features['x_coord']))
    max_y_coord = int(max(features['y_coord']))

    coordinates = [(x, y) for x in range(max_x_coord) for y in range(max_y_coord)]
    coordinates = pd.DataFrame.from_records(coordinates, columns=['x_coord', 'y_coord'])

    features_img = pd.merge(left=coordinates,
                            right=features[[x, y, rock_presence_column, 'id', 'x_coord', 'y_coord']],
                            on=('x_coord', 'y_coord'), how='outer')

    x = features_img['x_coord']
    y = features_img['y_coord']

    # empty arrays for the value and ids
    value_array = np.nan * np.empty((max_x_coord + 1, max_y_coord + 1))
    id_array = np.nan * np.empty((max_x_coord + 1, max_y_coord + 1))

    value_array[x, y] = features_img[rock_presence_column]
    id_array[x, y] = features_img['id']

    value_array = np.nan_to_num(x=value_array, nan=2)

    # list of region labels
    region_labels = label(value_array, background=2)

    region_labels_col = region_labels.flatten()
    id_flattened_column = id_array.flatten()

    # add region labels to original dataframe
    regions_df = pd.DataFrame(data={'region_label': region_labels_col, 'point_id': id_flattened_column})
    regions_df = pd.merge(left=regions_df, right=features, left_on='point_id', right_on='id', how='right')

    # Find where variable of interest == 0 and replace with random numbers to group data
    mask = regions_df[rock_presence_column] == 0.0
    random_col = np.random.randint(np.max(regions_df['region_label']),
                                   len(regions_df['region_label']), len(mask))

    regions_df['region_label'] = regions_df['region_label'].\
        mask(regions_df[rock_presence_column] == 0, random_col)

    regions_df = regions_df.drop(['id', 'point_id', 'x_coord', 'y_coord'], axis=1)  # Drop unneeded cols

    return regions_df


def t_t_split(features, test_size, rock_presence_column, test_catchment, contiguous_split=False,
              regional_split=False):
    """Returns pandas.Dataframes

    Splits data into testing and training using different methods as defined by user
    If using in regional_split mode, expects a column called "catchment" and "test_catchment" to 
    be a value of some  of the rows in this column in the features dataset
    
    Variables
        features - input dataframe of features (x,y, rock_presence_column etc.)
        test_size - 
        rock_presence_column - (string) name of dependent variable column
        test_catchment - 
        contiguous_split - boolean as to whether to split considering/discarding contiguous points
        regional_split - splits data based on an expected column titled "catchment" - requires 
            'test_catchment' to also be set
            - catchment column == test_catchment, assigned as test dataset
            - catchment column != test_catchment, assigned as train dataset
        test_catchment : where "test_catchment" is set, this is used to then split the dataframe 
            into testing and training sets   
            based on the "catchment" column
              e.g. test_catchment set to "aoi" - results in entires where catchment column=="aoi" 
                   being set as training data and where catchment column!="aoi" being set as test
    
    included here so same t-t datasets can be generated by different scripts (e.g. for plotting)
    uses the region column to preserve contiguous points in the splitting, subsequently drops this 
    column
    """
    
    if contiguous_split:

        rock_present_rows = features[features[rock_presence_column] == 1]
        max_present_region_value = max(rock_present_rows.region_label)
        present_region_values = range(max_present_region_value)
        test_present_regions = random.sample(present_region_values, int(test_size * max_present_region_value))

        # split rock absent rows into train and test
        rock_absent_rows = features[features[rock_presence_column] == 0]
        max_absent_region_value = max(rock_absent_rows.region_label)
        absent_region_values = range(max_present_region_value + 1, max_absent_region_value)
        test_absent_regions = random.sample(absent_region_values, int(test_size * len(absent_region_values)))

        test_regions = test_present_regions + test_absent_regions

        test_features = features[features['region_label'].isin(test_regions)].drop('region_label', axis=1)
        train_features = features[~features['region_label'].isin(test_regions)].drop('region_label', axis=1)

        test_labels = test_features[rock_presence_column]
        train_labels = train_features[rock_presence_column]

    elif regional_split:

        test_features = features[features['catchment'] == test_catchment]
        train_features = features[features['catchment'] != test_catchment]

        test_labels = test_features[rock_presence_column]
        train_labels = train_features[rock_presence_column]

    else:
        train_features, test_features = train_test_split(features, test_size=test_size)

        test_labels = test_features[rock_presence_column]
        train_labels = train_features[rock_presence_column]

    return train_features, test_features, train_labels, test_labels


def class_rescale(df, scale_method, rock_presence_column):
    """ Returns pandas.DataFrame

    Variables:
        df - input pandas.DataFrame
        scale_method - 'upsample'/'downsample' (sklearn.utils.resample())
        rock_presence_column - (string) name of dependent variable column

    Rescales test set based on variable column to include 
    equal number of each class by up or downscaling
    """
    pos_instances = df[df[rock_presence_column] == 1]
    neg_instances = df[df[rock_presence_column] == 0]

    if len(pos_instances) > len(neg_instances):
        maj_class = pos_instances
        min_class = neg_instances
    else:
        maj_class = neg_instances
        min_class = pos_instances

    if scale_method == 'upsample':
        min_class = resample(min_class,
                             replace=True,
                             n_samples=len(maj_class),
                             random_state=57)

    elif scale_method == 'downsample':
        maj_class = resample(maj_class,
                             replace=False,
                             n_samples=len(min_class),
                             random_state=57)

    df = pd.concat([min_class, maj_class])

    return df


def extract_from_dir(data_path, filename_identifier, rock_presence_column,
                     directional_variables, one_hot_encode, rescale,
                     interval, test_prop, train_data_drops, test_scale_method,
                     train_scale_method, split_method, test_catchment=None):
    """ Returns pandas.DataFrames

    Trawls a directory for matching csv files and performs csv processing to output clean dataset
    
    Variables
        test_prop - proportion of data to set aside for testing (value between 0.0 and 1.0)
        split_method - if set to '', will split data using the proportion defined by the 
                        test_prop variable
                     - if set to 'regional', will split data using the value set for test_catchment 
                        which would be expected to occur under a column entitle 'catchment'   
        data_path - path of input dataset
        filename_identifier - wildcard statement by which to ID relevant files e.g. "*.csv"
        rock_presence_column - (string) name of dependent variable column
        directional_variables - names of directional variables as named in the input file as column headers e.g. "aspect"
        one_hot_encode - BOOLEAN - if true applied pd.get_dummies() to input data
        rescale - BOOLEAN - if true, applies rescale_data() to data
        interval - indiciative of spacing between points (if xy positions are cell centres of a 10x10m grid, interval = 10) 
        train_data_drops - columns to drop from input dataset for training (as per column names)
        test_scale_method - 'upsample'/'downsample' (sklearn.utils.resample())
        train_scale_method - 'upsample'/'downsample' (sklearn.utils.resample())
        test_catchment - where "test_catchment" is set, this is used to then split the dataframe 
            into testing and training sets based on the "catchment" column
            e.g. test_catchment set to "aoi" - results in entires where catchment column=="aoi"
            being set as training data and where catchment column!="aoi" being set as test
    
    NOTE: THIS WILL USE ALL FILES IN DATA_PATH THAT MATCH FILENAME_IDENTIFIER
    
    Gets all data from `data_path` whilst considering `filename_identifier` 
        * uses `os.walk(data_path)` and `fnmatch.filter(filenames, filename_identifier)`
    """
    matches = []
    for root, dirnames, filenames in os.walk(data_path):
        for filename in fnmatch.filter(filenames, filename_identifier):
            matches.append(os.path.join(root, filename))

    train_df_out = pd.DataFrame()
    test_df_out = pd.DataFrame()

    for match in matches:

        features = pd.read_csv(match)

        # add column to show catchment
        features['catchment'] = os.path.basename(match)

        # Brute force drop any rows with nan
        features = features.dropna(axis=0, how='any')

        features = drop_invalid_results(features, rock_presence_column)

        # Apply this to dataset
        features_original = features  # keep for later - not hot-encoded

        # retype compass point variables
        for f in directional_variables:

            try:
                features[f] = categorize_aspect(features[f])

            except:
                logging.info("check directional variables are entered correctly")

        if one_hot_encode:
            features = pd.get_dummies(features)
            # Display the first 5 rows of the last 12 columns
            # print(features.iloc[:,5:].head(5))
            print('The shape of our features before hot-encoding is:', features_original.shape)
            print('The shape of our features after hot-encoding is:', features.shape)

        # Rescaling
        features[rock_presence_column] = features[rock_presence_column].astype('category')

        if rescale:
            features = rescale_data(features, rock_presence_column)

        # Train Test Split

        # Saving feature names for later use
        # feature_list = list(features.columns)
        # feature_types = features.dtypes

        # add region labels
        features = label_contiguous_points(features, interval, rock_presence_column, 'x', 'y')

        # Split the data into training and testing sets
        if split_method == 'regional':
            train_features, test_features, train_labels, test_labels = t_t_split(features, test_prop,
                                                                                 rock_presence_column, test_catchment,
                                                                                 False, True)
        else:
            train_features, test_features, train_labels, test_labels = t_t_split(features, test_prop,
                                                                                 rock_presence_column, None, False,
                                                                                 False)

        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)

        assert train_features.shape[0] != test_features.shape[0]  # should be different
        assert train_features.shape[0] == train_labels.shape[0]  # should be the same length (1 label for each row)

        train = train_features
        test = test_features

        for f in train_data_drops:
            train = train.drop(f, axis=1)

        # class rescaling
        test = class_rescale(test, test_scale_method, rock_presence_column)
        train = class_rescale(train, train_scale_method, rock_presence_column)

        # append to output df
        train_df_out = pd.concat([train_df_out, train])
        test_df_out = pd.concat([test_df_out, test])

        # drop 'region_label' column (only required for splitting data based on contiguous points)
        train_df_out=train_df_out.drop(['region_label'], axis=1)
        test_df_out=test_df_out.drop(['region_label'], axis=1)

    return train_df_out, test_df_out