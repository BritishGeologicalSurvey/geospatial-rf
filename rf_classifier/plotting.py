# -*- coding: utf-8 -*-
"""
Basic dataframe plotting helper functions

@author: ahall
"""

import matplotlib.pyplot as plt

def plot_data_points(df1, df2, x, y, label1, label2, title):
    """Return a plot in the current session

    Plots 2 dataframes considering coordinates [x,y], labels and titles

    Variables:
        df1 - first dataframe to plot
        df2 - secomnd dataframe to plot
        x - x variable column
        y - y variable column
        label1 - labels to apply to df1 data  
        label2 - labels to apply to df2 data
        title - plot title
    """
    plt.figure(figsize=(16, 12), dpi=80)
    plt.scatter(df1[x], df1[y], c='b', s=2, label=label1)
    plt.scatter(df2[x], df2[y], c='r', s=2, label=label2)
    plt.title(title)
    plt.legend()
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.show()
