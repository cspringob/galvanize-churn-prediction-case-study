from sklearn.datasets import load_iris
iris=load_iris()

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

from pipeline import Pipeline

def adarefit(model, df_data, df_target):
    """
    INPUTS:
           model: adaBoost model with optimized hyperparameters
           df_data : dataframe with predictor data
           df_target: dataframe with y labels
    OUTPUTS:
           model: adaBoost model re-fitted on new data
           feature_importance: Zip of feature importances and feature  names."""
    X, xcols, y = _extract_cols(df_data, df_target)
    model = model.fit(X, y)
    feature_importance = zip(xcols, model.feature_importances_)
    return model, feature_importance

def logreg(df_data, df_target):
    parameters = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    X, xcols, y = _extract_cols(df_data, df_target)
    pipe = Pipeline(LogisticRegression,X, y, parameters)
    pipe.grid_search()
    model = pipe.fit_model
    feature_importance = zip(xcols, model.coef_[0])
    return pipe, feature_importance

def adapipe(df_data, df_target):
    """
    INPUTS:
           df_data : dataframe with predictor data
           df_target : dataframe with y labels
    OUTPUTS:
           model: adaBoost model with grid searched model
           feature_importance: Zip of feature importances and feature  names."""
    parameters = {'n_estimators' : [i for i in range(35,75,5)],
                  'learning_rate' : [.2]}
    X, xcols, y = _extract_cols(df_data, df_target)
    pipe = Pipeline(AdaBoostClassifier,X, y, parameters)
    pipe.grid_search()
    model = pipe.fit_model
    feature_importance = zip(xcols, model.feature_importances_)
    return pipe, feature_importance

def _extract_cols(features, targets):
    ''' extract feature values, feature columns and target values, in that order.'''
    return features.values, features.columns, targets.values
