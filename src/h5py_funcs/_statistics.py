import numpy as np
import scipy as sp
import scipy.stats as sp_stats
import pandas as pd
import statsmodels.api as sm


class univariate_statistics:
    def __init__(self, data: pd.DataFrame, columns: None|str|list[str] = None):
        if columns is not None:
            self.data = data[columns]
        else:
            self.data = data
        
        self.statistics = {}

    def compute_all_statistics(self, columns: None|str|list[str] = None, max_nth_moment:int=10) -> None:
        self.statistics['nth_moment_biased'] = {n: _nth_moment(self.data, n=n, axis=0) for n in range(1, max_nth_moment+1)}
        self.statistics['nth_moment_unbiased'] = {n: _nth_moment(self.data, n=n-1, axis=0) for n in range(1, max_nth_moment+1)}
        self.statistics['nth_lmoment'] = {n: _nth_lmoment(self.data, n=n, axis=0) for n in range(1, max_nth_moment+1)}
        self.statistics['gmean'] = self.get_gmean(columns)
        self.statistics['hmean'] = self.get_hmean(columns)
        self.statistics['mean'] = self.get_mean(columns)
        self.statistics['variance'] = self.get_variance(columns)
        self.statistics['std'] = self.get_std(columns)
        self.statistics['median'] = self.get_median(columns)
        self.statistics['skewness'] = self.get_skewness(columns)
        self.statistics['kurtosis'] = self.get_kurtosis(columns)
        self.statistics['entropy'] = self.get_entropy(columns)
        self.statistics['cross_entropy'] = self.get_cross_entropy(columns)

    def get_mean(self, columns: None|str|list[str] = None) -> list[float]:
        if columns is None:
            return self.data.mean(axis=0)
        elif isinstance(columns, str):
            return [self.data[columns].mean(axis=0)]
        elif isinstance(columns, list):
            return self.data.mean(axis=0)

    def get_variance(self, columns: None|str|list[str] = None) -> list[float]:
        if columns is None:
            return self.data.var(axis=0)
        elif isinstance(columns, str):
            return [self.data[columns].var(axis=0)]
        elif isinstance(columns, list):
            return self.data.var(axis=0)


    def get_std(self, columns: None|str|list[str] = None) -> list[float]:
        if columns is None:
            return self.data.std(axis=0)
        elif isinstance(columns, str):
            return [self.data[columns].std(axis=0)]
        elif isinstance(columns, list):
            return self.data.std(axis=0)
        

    def get_median(self, columns: None|str|list[str] = None) -> list[float]:
        if columns is None:
            return self.data.median(axis=0)
        elif isinstance(columns, str):
            return [self.data[columns].median(axis=0)]
        elif isinstance(columns, list):
            return self.data.median(axis=0)

    def get_skewness(self, columns: None|str|list[str] = None) -> list[float]:
        if columns is None:
            return self.data.skew(axis=0)
        elif isinstance(columns, str):
            return [self.data[columns].skew(axis=0)]
        elif isinstance(columns, list):
            return self.data.skew(axis=0)

    def get_kurtosis(self, columns: None|str|list[str] = None) -> list[float]:
        if columns is None:
            return self.data.kurtosis(axis=0)
        elif isinstance(columns, str):
            return [self.data[columns].kurtosis(axis=0)]
        elif isinstance(columns, list):
            return self.data.kurtosis(axis=0)
        
    def get_gmean(self, columns: None|str|list[str] = None) -> list[float]:
        if columns is None:
            return sp_stats.gmean(self.data, axis=0)
        elif isinstance(columns, str):
            return [sp_stats.gmean(self.data[columns], axis=0)]
        elif isinstance(columns, list):
            return sp_stats.gmean(self.data[columns], axis=0)
        
    def get_hmean(self, columns: None|str|list[str] = None) -> list[float]:
        if columns is None:
            return sp_stats.hmean(self.data, axis=0)
        elif isinstance(columns, str):
            return [sp_stats.hmean(self.data[columns], axis=0)]
        elif isinstance(columns, list):
            return sp_stats.hmean(self.data[columns], axis=0)
    
    def get_entropy(self, columns: None|str|list[str] = None) -> list[float]:
        """
        Computes the entropy of the data.
        :param columns: The columns to compute the entropy for. If None, computes for all columns.
        :return: The entropy of the data.
        """
        if columns is None:
            return sp_stats.entropy(self.data, axis=0)
        elif isinstance(columns, str):
            return [sp_stats.entropy(self.data, axis=0)]
        elif isinstance(columns, list):
            return sp_stats.entropy(self.data, axis=0)
        
    def get_cross_entropy(self, columns: None|list[str] = None) -> list[float]:
        """
        Computes the cross entropy of the data.
        :param columns: The columns to compute the cross entropy for. If None, computes for all columns.
        :return: The cross entropy of the data.
        """
        if columns is None:
            columns = self.data.columns
        return [[sp_stats.entropy(pk=self.data[i], qk=self.data[j], axis=0) for j in columns] for i in columns]
        

def _nth_moment(data: pd.Series|pd.DataFrame, n: int, axis:int=0) -> list[float]:
    """
    Computes the nth moment(s) of the data.
    :param data: The data to compute the moment for.
    :param n: The order of the moment.
    :param axis: The axis along which to compute the moment. Defaults to 0.
    :return: The nth moment of the data.
    """
    moments = sp_stats.moment(data, moment=n, axis=axis)
    if isinstance(data, pd.Series):
        return [moments]
    elif isinstance(data, pd.DataFrame):
        return moments
    
def _nth_lmoment(data: pd.Series|pd.DataFrame, n: int, axis:int=0) -> list[float]:
    """
    Computes the nth moment(s) of the data.
    :param data: The data to compute the moment for.
    :param n: The order of the moment.
    :param axis: The axis along which to compute the moment. Defaults to 0.
    :return: The nth moment of the data.
    """
    moments = sp_stats.lmoment(data, order=n, axis=axis, standardize=False)
    if isinstance(data, pd.Series):
        return [moments]
    elif isinstance(data, pd.DataFrame):
        return moments