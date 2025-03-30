import logging
from time import time
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from freqtrade.freqai.base_models.BaseIsolationModel import BaseIsolationModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

from freqtrade.plot.plotting import go, make_subplots, store_plot_file

import shap
import matplotlib.pyplot as plt
import os


logger = logging.getLogger(__name__)


class SKLearnIsolationForest(BaseIsolationModel):
    """
    Base class for regression type models (e.g. Catboost, LightGBM, XGboost etc.).
    User *must* inherit from this class and set fit(). See example scripts
    such as prediction_models/CatboostClassifier.py for guidance.
    """     

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        model = IsolationForest(**self.model_training_parameters)

        model.fit(X)

        eval_scores = model.decision_function(X)
        logger.info("Score: %s", eval_scores)

        # Use shap to plot_feature_importance method after training the model
        # self.plot_feature_importance(model, dk, X)
        
        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param  unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """
        (pred_df, dk.do_predict) = super().predict(unfiltered_df, dk, **kwargs)

        return (pred_df, dk.do_predict)