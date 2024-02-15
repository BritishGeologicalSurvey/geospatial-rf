# -*- coding: utf-8 -*-
"""
H20 frame creation functions for model set up 

@author: ahall
"""
import h2o
from h2o.estimators import H2ORandomForestEstimator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def create_h2o_sets(train, test, dependant_column):
    """Return H20Frames 
    
    Variables:
      train - pandas.DataFrame of train data
      test - pandas.DataFrame of test data
      dependant_column - (string) name of dependent variable column

    Converts train and test set to h2o frames
    """
    hf_train = h2o.H2OFrame(train)
    hf_test = h2o.H2OFrame(test)
    
    hf_train[dependant_column] = hf_train[dependant_column].asfactor()
    hf_test[dependant_column] = hf_test[dependant_column].asfactor()
    
    return hf_train, hf_test


def train_h2o_rf(hf_train, hf_test, params, dependant_column, ignore=['catchment']):
    """Return h2o.estimators.H2ORandomForestEstimator trained model object

    Trains up a random forest for mixed data and initializes a h2o instance

    Variables:
      hf_train - H2O train dataframe (see create_h2o_sets()
      hf_test - H2O test dataframe (see create_h2o_sets()
      params - dictionary of params to consider for model estimation
        e.g. RF_PARAMS = {"ntrees": 500,
                          "max_depth": 50, 
                          "min_rows": 20,
                          "mtries": -1, #default - samples sqrt number of columns (-1)
                          "calibrate_model": False,
                          "binomial_double_trees": True,#essentially doubles tree count
                          "balance_classes": False, #weight predictions in order to compensate for class imbalance
                          "histogram_type": "Random" #use extrememly randomized trees (remove argument to change)
                          }
      dependant_column - dependent column name
      ignore - columns to ignore 

    """

    # Build and train the model:
    # Hyperparameter choice notes:
    # max_depth: default of 20 is too simple - all trees end up at max depth in that case
    # binomial_double_trees: essentially doubles tree count
    # balance_classes: weight predictions in order to compensate for class imbalance
    # histogram_type: use extrememly randomized trees (remove argument to change)
    rf = H2ORandomForestEstimator(ntrees=params['ntrees'],
                                  max_depth=params['max_depth'],
                                  min_rows=params['min_rows'],
                                  calibrate_model=params['calibrate_model'],
                                  binomial_double_trees=params['binomial_double_trees'],
                                  balance_classes=params['balance_classes'],
                                  histogram_type=params['histogram_type']
                                  )

    rf.train(y=dependant_column,
             training_frame=hf_train,
             ignored_columns=ignore)
    
    return rf


def get_classification_report(pred, hf_test, dependant_column):
    """Returns dataframes of predicted and input classifiation values

    Outputs sklearn classification report from the test set (h2o_frame format and predictions)
    """
    
    y_pred = pred['predict'].as_data_frame()
    y_true = hf_test[dependant_column].as_data_frame()
    
    print(classification_report(y_true, y_pred))
    
    return y_pred, y_true


def plot_confusion_matrix(y_true, y_pred, title=None):
    """Plot confusion matrix in current session"""
    cm = confusion_matrix(y_true, y_pred)
    p = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    p.set_title(title)
    p.set_xlabel('predicted')
    p.set_ylabel('actual')
