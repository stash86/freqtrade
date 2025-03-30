import logging
from time import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel

logger = logging.getLogger(__name__)


class BaseIsolationModel(IFreqaiModel):
    """
    Base class for anomaly detection models (e.g. Isolation Forest, One-Class SVM, etc.).
    User *must* inherit from this class and set the `fit()` method. This class
    simplifies the use of anomaly detection algorithms within the Frequency AI framework.
    """
    
    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_df: Full dataframe for the current training period
        :param metadata: pair metadata from strategy.
        :return:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info(f"-------------------- Starting training {pair} --------------------")

        start_time = time()

        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        start_date = unfiltered_df["date"].iloc[0].strftime("%Y-%m-%d")
        end_date = unfiltered_df["date"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(f"-------------------- Training on data from {start_date} to "
                    f"{end_date} --------------------")

        # split data into train/test data.
        dd = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if not self.freqai_info.get("fit_live_predictions_candles", 0) or not self.live:
            dk.fit_labels()
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)

        (dd["train_features"],
         dd["train_labels"],
         dd["train_weights"]) = dk.feature_pipeline.fit_transform(dd["train_features"],
                                                                  dd["train_labels"],
                                                                  dd["train_weights"])

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            (dd["test_features"],
             dd["test_labels"],
             dd["test_weights"]) = dk.feature_pipeline.transform(dd["test_features"],
                                                                 dd["test_labels"],
                                                                 dd["test_weights"])
        logger.info(
            f"Training model on {len(dk.data_dictionary['train_features'].columns)} features"
        )
        logger.info(f"Training model on {len(dd['train_features'])} data points")

        model = self.fit(dd, dk)

        end_time = time()

        logger.info(f"-------------------- Done training {pair} "
                    f"({end_time - start_time:.2f} secs) --------------------")

        return model
    

    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Use the fitted model to predict anomalies.
        :param unfiltered_df: Full dataframe.
        :param dk: Datakitchen object for feature engineering and preprocessing.
        :return: DataFrame containing the predictions and np.array of 1s and 0s to indicate anomalies.
        """
        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
    
        dk.data_dictionary["prediction_features"] = filtered_df

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True)
    
         # In anomaly detection models, 'predict' typically identifies outliers
        predictions = self.model.predict(dk.data_dictionary["prediction_features"])

        # Outliers are typically labeled as -1, so we convert to a binary indicator (0 for inliers, 1 for outliers)
        predictions = np.where(predictions == -1, 1, 0)
        
        # Obtain anomaly scores using decision_function
        anomaly_scores = self.model.decision_function(dk.data_dictionary["prediction_features"])
        
        # Make all anomaly scores positive
        anomaly_scores_positive = -anomaly_scores + abs(anomaly_scores.min())
        
        # Scale adjusted scores to a 0-1 "pseudo-probability" range
        scaler = MinMaxScaler(feature_range=(0, 1))
        pseudo_probas = scaler.fit_transform(anomaly_scores_positive.reshape(-1, 1))
        
        # Add anomaly scores and pseudo probabilities to your predictions dataframe
        predictions = predictions.ravel()
        anomaly_scores = anomaly_scores.ravel()
        pseudo_probas = pseudo_probas.ravel()
        
        # Prepare the anomaly data in a dictionary
        data = {
            dk.label_list[0]: predictions,
            dk.label_list[1]: anomaly_scores,
            dk.label_list[2]: pseudo_probas,
        }

        # Create a DataFrame to hold the anomaly predictions
        pred_df = DataFrame(data, columns=dk.label_list)
                
        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(len(pred_df))
        
        # Indicate where predictions were made
        dk.do_predict = np.ones_like(predictions, dtype=np.int_)
                
        return (pred_df, dk.do_predict)