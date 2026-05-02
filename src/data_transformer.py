# src/data_transformer.py
from pathlib import Path
from typing import Optional, Union
import pandas as pd

class DataTransformer:
    COL_MAP = {
        "光伏_日前(T)": "PV_DayAhead_T",
        "省调负荷_日前(T)": "ProvincialLoad_DayAhead_T",
        "竞价空间_日前(T)": "BiddingSpace_DayAhead_T",
        "联络线计划日_日前(T)": "InterconnectionPlan_DayAhead_T",
        "非市场化机组出力_日前(T)": "NonMarketUnitOutput_DayAhead_T",
        "风力_日前(T)": "Wind_DayAhead_T",
        "光伏_实时(T-2)": "PV_RealTime_T_minus_2",
        "省调负荷_实时(T-2)": "ProvincialLoad_RealTime_T_minus_2",
        "竞价空间_实时(T-2)": "BiddingSpace_RealTime_T_minus_2",
        "联络线计划日_实时(T-2)": "Interconnection_RealTime_T_minus_2",
        "非市场化机组出力_实时(T-2)": "NonMarketUnitOutput_RealTime_T_minus_2",
        "风力_实时(T-2)": "Wind_RealTime_T_minus_2",
        "出清价格_实时(T-2)": "ClearingPrice_RealTime_T_minus_2",
        "出清价格_日前(T-1)": "ClearingPrice_DayAhead_T_minus_1",
    }

    LABEL_MAP = {
        "搏低价": "target",
    }

    def __init__(self, column_map=None, label_map=None):
        self.column_map = column_map or self.COL_MAP
        self.label_map = label_map or self.LABEL_MAP

    def load_features(self, data_path):
        """
        Load feature data from CSV, rename columns, parse datetime index,
        and sort chronologically.
        """
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df = df.rename(columns=self.column_map)
        df = df.sort_index()
        return df

    def load_labels(self, labels_path):
        """
        Load labels from CSV and rename target column.
        """
        labels = pd.read_csv(labels_path, index_col=0, parse_dates=True)
        labels = labels.rename(columns=self.label_map)
        labels = labels.sort_index()
        return labels

    def load_dataset(self, data_path, labels_path=None):
        """
        Load either training data or prediction data.

        If labels_path is provided:
            returns dataframe with target column.

        If labels_path is None:
            returns dataframe without target column.
        """
        df = self.load_features(data_path)

        if labels_path is not None:
            labels = self.load_labels(labels_path)
            df = df.join(labels["target"], how="inner")

        return df