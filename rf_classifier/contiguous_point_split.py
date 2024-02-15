# -*- coding: utf-8 -*-
"""
Functions to identify and label contiguous spatial points

@author: ahall
"""
import pandas as pd
from skimage.measure import label
import numpy as np

def label_contiguous_points(features, interval=10):
    """Returns pandas.DataFrame

    Generates dataframe of labelled contiguous points based on an input array

    Variables:
        features - pandas.DataFrame of spatial features 
            - ***Further development required***
            - Input features expected (and hard-encoded) as features[['x', 'y', 'value', 'id']]
        interval - spatial interval of x and y coordimates in same units as position coordinmates (default = 10 m)
    """
    features['id'] = range(1, len(features) + 1)
    points_cols = features[['x', 'y', 'value', 'id']]
    min_x = min(features['x'])
    # max_x = max(features['x'])
    min_y = min(features['y'])
    # max_y = max(features['y'])
    #interval = 10
    points_cols['x_coord'] = (features['x'] - min_x)/interval
    points_cols['y_coord'] = (features['y'] - min_y)/interval
    
    max_x_coord = int(max(points_cols['x_coord']))
    max_y_coord = int(max(points_cols['y_coord']))
    
    coordinates = [(x, y) for x in range(max_x_coord) for y in range(max_y_coord)]
    
    coordinates = pd.DataFrame.from_records(coordinates, columns=['x_coord', 'y_coord'])
    
    features_img = pd.merge(left=coordinates, right=points_cols, on=('x_coord', 'y_coord'), how='outer')
    
    x = features_img['x_coord']
    y = features_img['y_coord']
    
    # empty arrays for the value and ids
    value_array = np.nan * np.empty((max_x_coord+1, max_y_coord+1))
    id_array = np.nan * np.empty((max_x_coord+1, max_y_coord+1))
    
    value_array[x, y] = features_img['value']
    id_array[x, y] = features_img['id']
    
    value_array = np.nan_to_num(x=value_array, nan=2)
    
    # list of region labels
    region_labels = label(value_array, background=2)
    
    region_labels_col = region_labels.flatten()
    id_flattened_column = id_array.flatten()
    
    # add region labels to original dataframe
    regions_df = pd.DataFrame(data={'region_label': region_labels_col, 'point_id': id_flattened_column})
    regions_df = pd.merge(left=regions_df, right=features, left_on='point_id', right_on='id', how='right')
    
    # give 0-value (ie no rock) rows a region identifier but make it a random number so each point has a unique
    # region identifier - ie they will be split randomly between train and test
    mask = regions_df['value'] == 0
    regions_df['region_label'][mask] = np.random.randint(np.max(regions_df['region_label']),
                                                         len(regions_df['region_label']), len(mask))
    
    regions_df = regions_df.drop(['point_id', 'id'], axis=1)
    
    return regions_df
