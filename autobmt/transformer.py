#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: transformer.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-03-31
'''

import copy
import math
from functools import wraps

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin

from .statistics import probability, WOE
from .utils import split_points_to_bin, FILLNA, save_json, split_target

DEFAULT_NAME = 'default_name'
EMPTY_BIN = -1
ELSE_GROUP = 'else'


def df_exclude_cols(func):
    @wraps(func)
    def exclude_cols(self, X, y, **kwargs):
        exclude = kwargs.get('exclude', None)
        if exclude is not None and isinstance(X, pd.DataFrame):
            X = X.drop(columns=exclude)
            del kwargs['exclude']

        return func(self, X, y, **kwargs)

    return exclude_cols


def df_select_dtypes(func):
    @wraps(func)
    def select_dtypes(self, X, y, **kwargs):
        select_dtypes = kwargs.get('select_dtypes', None)
        if select_dtypes is not None and isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include=select_dtypes)
            del kwargs['select_dtypes']

        return func(self, X, y, **kwargs)

    return select_dtypes


def _check_duplicated_keys(X):
    if isinstance(X, pd.DataFrame) and X.columns.has_duplicates:
        keys = X.columns[X.columns.duplicated()].values
        raise Exception("X has duplicate keys `{keys}`".format(keys=str(keys)))

    return True


class FeatureBin(TransformerMixin):
    """
    分箱类
    """

    def __init__(self, n_jobs=-1):
        self.splits_dict = dict()
        self.n_jobs = n_jobs

    def __len__(self):
        return len(self.splits_dict.keys())

    def __contains__(self, key):
        return key in self.splits_dict

    def __getitem__(self, key):
        return self.splits_dict[key]

    def __setitem__(self, key, value):
        self.splits_dict[key] = value

    def __iter__(self):
        return iter(self.splits_dict)

    @df_exclude_cols
    @df_select_dtypes
    def fit(self, X, y, **kwargs):
        """
        分箱
        Args:
            X (DataFrame|array-like): 要分箱的X
            y (str|array-like): 目标变量
            min_sample_rate (number) : 每个箱子的最小占比，默认0.05
            n_bins (int): 需要分成几箱，默认10
            is_need_monotonic (bool) : 是否强制单调，默认True，强制单调
            is_empty_bin (bool) : 是否将空箱单独归为一个箱子，默认True，空箱单独归1箱
            min_threshold (number): 最小的卡方阀值
            exclude (str|array-like): 排除的特征，该特征将不参与分箱
            select_dtypes (str|numpy.dtypes): `'object'`, `'number'` 等. 只有选定的数据类型才会被分箱

        """

        # assert y.isin([0, 1]).all(), 'ERROR: :-) :-) 目标变量不是0/1值，请检查！！！'

        if not isinstance(X, pd.DataFrame):
            fea_name, splits = self._fit(X, y, **kwargs)
            self.splits_dict[fea_name] = splits
            return self

        if isinstance(y, str):
            # y = X.pop(y)
            X, y = split_target(X, y)

        _check_duplicated_keys(X)

        data = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit)(X[col], y, **kwargs) for col in X)  # 批量处理

        self.splits_dict = dict(data)

        return self

    def _fit(self, X, y, method='chi', is_empty_bin=True, **kwargs):  # method='dt'修改为method='chi'
        """
        分箱
        Args:
            X (DataFrame|array-like): 要分箱的X
            y (str|array-like): 目标变量
            min_sample_rate (number) : 每个箱子的最小占比
            n_bins (int): 需要分成几箱
            is_need_monotonic (bool) : 是否强制单调
            is_empty_bin (bool) : 是否将空箱单独归为一个箱子
            min_threshold (number): 最小的卡方阀值

        Returns:
            str : 分箱变量名
            array : 分割点

        """

        from .feature_binning import bin_method_run

        fea_name = DEFAULT_NAME
        if hasattr(X, 'name'):
            fea_name = X.name

        # 判断是否连续型变量，如果不是，将其原始值替换为0、1、2、3这样有序的连续性变量，序是按原始值所对应的woe值大小来给予的
        unique_X_val = None
        if not np.issubdtype(X.dtype, np.number):
            transer = WoeTransformer()
            # if X.dtype.type is np.object_:
            #     X = X.astype(np.str)
            empty_mask = pd.isna(X).any()
            if empty_mask:
                X = X.astype(np.str)
            woe = transer.fit_transform(X, y)
            # 获取变量的唯一值，及其所在该变量中的索引；unique_X_val=['A' 'B' 'C' 'D' 'E' 'F' 'G'] unique_X_index=[25  0  2  9  1  6 18]
            unique_X_val, unique_X_index = np.unique(X, return_index=True)
            # 通过原始值所在的索引将其原始对应的woe值取出。unique_woe=[-0.10178269 -0.44183275  0.22730944  0.15707894  0.50662326 -0.27946387 -0.10178269]
            unique_woe = woe[unique_X_index]
            # argsort函数是将woe值从小到大排序，然后将排序后的woe值所对应的原始woe值所在的索引输出；woe_argsort_index=[1 5 0 6 3 2 4]
            woe_argsort_index = np.argsort(unique_woe)
            # 变量唯一值按woe值从小到大的顺序调整变量唯一值的位置，变成有顺序的；unique_X_val=['B' 'F' 'A' 'G' 'D' 'C' 'E']
            unique_X_val = unique_X_val[woe_argsort_index]
            # 将原始X根据unique_X_val=['B' 'F' 'A' 'G' 'D' 'C' 'E']的有序顺序替换为0、1、2、3
            # unique_X_val=['B' 'F' 'G' 'nan' 'D' 'C' 'E']
            if empty_mask and is_empty_bin:
                unique_X_val = unique_X_val[np.isin(unique_X_val, 'nan', invert=True)]
            X = self._raw_category_x_to_bin(X, unique_X_val, is_empty_bin=is_empty_bin)

        splits = bin_method_run(X, y, method, is_empty_bin=is_empty_bin, **kwargs)

        # 如果不是连续型变量，X原始值被0、1、2、3替换了，自然出来的splits也是数值，需要将splits中的数值从unique_X_val=['B' 'F' 'A' 'G' 'D' 'C' 'E']进行还原
        splits = self._restore_category_splits(splits, unique_X_val)

        return fea_name, splits

    def transform(self, X, bins_dict={}, **kwargs):
        """
        原始数据根据分割点变换原始X
        Args:
            X (DataFrame|array-like): 需要转换的原始X
            bins_dict: 分箱字典, 形如: {'D157': [-999, 1.0, 2.0, 3.0, 5.0, inf]}
            **kwargs:

        Returns:

        """

        if not isinstance(bins_dict, dict):
            assert '请传入类似 {\'D157\': [-999, 1.0, 2.0, 3.0, 5.0, inf]}'

        if not bins_dict:
            bins_dict = self.splits_dict
        else:
            bins_dict = {k: np.array(v) for k, v in bins_dict.items()}

        if getattr(X, 'ndim', 1) == 1:

            if hasattr(X, 'name'):  # pd.Series
                if X.name in bins_dict:
                    fea_name, bins = self._transform(X, bins_dict.get(X.name), **kwargs)
                    return bins
                else:
                    return X

            if len(bins_dict) == 1:
                if DEFAULT_NAME in bins_dict:
                    fea_name, bins = self._transform(X, bins_dict.get(DEFAULT_NAME), **kwargs)
                    return bins
                else:
                    return X

        # X.reset_index(inplace=True)

        _check_duplicated_keys(X)

        ###并行处理
        data_with_bins = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform)(X[col], bins, **kwargs) for col, bins in bins_dict.items() if col in X)
        ###并行处理

        if isinstance(X, dict):
            return dict(data_with_bins)
        else:
            bin_df = pd.DataFrame(dict(data_with_bins), index=X.index)
            X_cols = list(X.columns)
            no_bin_cols = list(set(X_cols) - set(bin_df.columns))
            bin_df[no_bin_cols] = X[no_bin_cols]
            # X.set_index('index', inplace=True)

            # return bin_df.set_index('index')
            return bin_df[X_cols]

    def _transform(self, X, splits, labels=False):
        """
        原始数据根据分割点变换原始X
        Args:
            X (DataFrame|array-like): 需要转换的原始X
            splits (array) : 分割点
            labels (bool) : 转换后的X是否需要标签，默认False，不需要

        Returns:

        """

        fea_name = DEFAULT_NAME
        if hasattr(X, 'name'):
            fea_name = X.name

        if splits.ndim > 1 or not np.issubdtype(splits.dtype, np.number):
            empty_mask = pd.isna(X).any()
            if empty_mask:
                X = X.astype(np.str)
            bins = self._raw_category_x_to_bin(X, splits)

        else:
            bins = np.zeros(len(X), dtype=int)

            if len(splits):  # TODO 需要看下splits为什么会有空
                if np.isnan(splits[-1]):
                    mask = pd.isna(X)
                    bins[~mask] = split_points_to_bin(X[~mask], splits[:-1])
                    bins[mask] = len(splits)
                else:
                    bins = split_points_to_bin(X, splits)

        if labels:
            splits_format = self.splits_point_format(
                splits,
                index=True)  ## ['0.[-inf, 0.7470300495624542)' '1.[0.7470300495624542, inf)' '2.nan']  ['0.B,F,G' '1.D,C,E' '2.nan']
            mask = (bins == EMPTY_BIN)
            bins = splits_format[bins]
            bins[mask] = FILLNA

        return fea_name, bins
        # row = pd.Series({fea_name:bins})
        # row.index = X.index
        # return row

    def _raw_category_x_to_bin(self, X, unique_val, is_empty_bin=False):
        """
        原始变量进行转换
        Args:
            X (array-like): 需要转换的X
            unique_val (array-like): 分割点
            is_empty_bin (bool): 是否有空箱

        Returns:

        """
        if is_empty_bin:
            bins = np.full(len(X), np.nan)
        else:
            bins = np.full(len(X), EMPTY_BIN)
            # bins = np.full(len(X), len(unique_val) - 1)

        for i in range(len(unique_val)):
            val = unique_val[i]
            if isinstance(val, str) and val == ELSE_GROUP:
                bins[bins == EMPTY_BIN] = i
            else:
                bins[np.isin(X, val)] = i
        return bins

    def _restore_category_splits(self, splits, x_val):
        """
        将分割点复原回原始值
        Args:
            splits (array-like): 分割点
            x_val (array-like): 原始分割点

        Returns:
            array: 原回原始值分割点
        """
        if x_val is None:
            return splits

        empty_mask = np.isnan(splits).any()
        if empty_mask:
            splits = splits[~np.isnan(splits)]

        if isinstance(x_val, np.ndarray):
            x_val = x_val.tolist()

        restore_category_splits = []
        start = 0
        for i in splits:
            index = math.ceil(i)
            restore_category_splits.append(x_val[start:index])
            start = index

        restore_category_splits.append(x_val[start:])

        if empty_mask:
            restore_category_splits.append(['nan'])

        return np.array(restore_category_splits)

    def splits_point_format(self, splits, index=False, ellipsis=None, decimal=None):
        """
        将分割点格式化，形如：[0.[4 ~ 7), 1.[7 ~ 10)]
        Args:
            splits (array-like): 分割点
            index (bool): 是否需要下标，0.[4 ~ 7)中的0
            ellipsis:

        Returns:
            array: 格式化后的分割点
        """
        ## 数值型：splits=[0.45343156        nan]，类别型：splits=[list(['B', 'F', 'G']) list(['D', 'C', 'E']) list(['nan'])]
        l = list()

        if not np.issubdtype(splits.dtype, np.number):
            # for i in splits:
            #     l.append(','.join(i))
            for i in splits:
                if isinstance(i, str) and i == ELSE_GROUP:
                    l.append(i)
                else:
                    label = ','.join(i)
                    if ellipsis is not None:
                        label = label[:ellipsis] + '..' if len(label) > ellipsis else label
                    l.append(label)

        else:
            is_empty_split = len(splits) > 0 and np.isnan(splits[-1])  # TODO 需要看下splits为什么会有空
            if is_empty_split:
                splits = splits[:-1]

            splits_ = [-np.inf] + splits.tolist() + [np.inf]
            for i in range(len(splits_) - 1):
                l.append('['.format(i) + str(splits_[i]) + ' ~ ' + str(splits_[i + 1]) + ')')

            if is_empty_split:
                l.append('nan')

        if index:
            l = ["{}.{}".format(i, v) for i, v in enumerate(l)]

        return np.array(l)

    def manual_bin(self, manual_set_dict):
        """
        手动分箱
        Args:
            manual_set_dict (dict|array-like): map结构的分箱点，形如: {'D157': [1.0, 2.0, 3.0, 5.0]}

        Returns:

        """
        if not isinstance(manual_set_dict, dict):
            manual_set_dict = {
                DEFAULT_NAME: manual_set_dict,
            }

        assert isinstance(manual_set_dict, dict), '请传入类似 {\'D157\': [1.0, 2.0, 3.0, 5.0]}'

        for i in manual_set_dict:
            self.splits_dict[i] = np.array(manual_set_dict[i])
        return self

    def export(self, to_dataframe=False, to_json=None, to_csv=None, bin_format=True, index=True):
        """
        导出规则到dict或json或csv文件
        Args:
            to_dataframe (bool): 是否导出成pd.DataFrame形式
            to_json (str): 保存成json的路径
            to_csv (str): 保存成csv的路径
            bin_format (bool): 是否将分割点格式化
            index (bool): 分割点格式化是否需要下标

        Returns:
            dict: 分割点规则字典

        """

        splits = copy.deepcopy(self.splits_dict)
        if bin_format:
            # splits = {k: self.splits_point_format(v, index=False).tolist() for k, v in splits.items()}
            splits = {k: list(self.splits_point_format(v, index=index)) for k, v in splits.items()}
        else:
            splits = {k: list(v.astype(float)) if np.issubdtype(v.dtype, np.number) else [','.join(i) for i in v] for
                      k, v in
                      splits.items()}
        if to_json is not None:
            save_json(splits, to_json)

        if to_dataframe or to_csv is not None:

            row = []
            for var_name in splits:
                for bin in splits[var_name]:
                    row.append({
                        'feature': var_name,
                        'bins': bin
                    })

            splits = pd.DataFrame(row)

        if to_csv is not None:
            splits.to_csv(to_csv, index=False)

        return splits


class WoeTransformer(TransformerMixin):
    """
    woe转换类
    """

    def __init__(self, n_jobs=-1):
        self.fea_woe_dict = dict()
        self.n_jobs = n_jobs

    def __len__(self):
        return len(self.fea_woe_dict.keys())

    def __contains__(self, key):
        return key in self.fea_woe_dict

    def __getitem__(self, key):
        return self.fea_woe_dict[key]

    def __setitem__(self, key, value):
        self.fea_woe_dict[key] = value

    def __iter__(self):
        return iter(self.fea_woe_dict)

    @df_exclude_cols
    @df_select_dtypes
    def fit(self, X, y):
        """
        woe转换
        Args:
            X (DataFrame|array-like): 需要转换woe的X
            y (str|array-like): 目标变量
            exclude (str|array-like): 排除的变量，该变量将不参与woe计算
            select_dtypes (str|numpy.dtypes): `'object'`, `'number'` 等. 只有选定的数据类型才会被计算

        Returns:

        """
        if not isinstance(X, pd.DataFrame):
            fea_name, value_woe = self._fit_woe(X, y)
            self.fea_woe_dict[fea_name] = value_woe
            return self

        if isinstance(y, str):
            # y = X.pop(y)
            X, y = split_target(X, y)

        _check_duplicated_keys(X)

        data = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_woe)(X[col], y) for col in X)  # 批量处理

        self.fea_woe_dict = dict(data)

        return self

    def _fit_woe(self, X, y):
        """
        woe转换
        Args:
            X (DataFrame|array-like): 需要计算woe的X
            y (str|array-like): 目标变量

        Returns:
            str : 变量名
            array : 计算出来的woe值

        """
        fea_name = DEFAULT_NAME
        if hasattr(X, 'name'):
            fea_name = X.name
        X = np.copy(X)  # Series, array 转 np.array
        if X.dtype.type is np.object_:
            X = X.astype(np.str)
        unique_val = np.unique(X)

        # # TODO 如果X是连续性变量，且有空。计算出来的woe不正确
        # if X.dtype.type is np.object_:
        #     X = X.astype(np.str)
        #     unique_val = np.unique(X)
        # else:
        #     unique_val = [int(i) for i in np.unique(X)]

        value_woe = dict()
        for val in unique_val:
            y_prob, n_prob = probability(y, mask=(X == val))  #
            value_woe[val] = WOE(y_prob, n_prob)

        return fea_name, value_woe

    def transform(self, X, fea_woe_dict={}, **kwargs):
        """
        将原始值用woe值替换
        Args:
            X (DataFrame|array-like): 需要转换woe的X
            fea_woe_dict (dict): 变量和woe值的字典，形如：{'D157': {0: -0.46554351769099783, 1: -0.10263802400162944}}
            **kwargs:

        Returns:
            DataFrame: 转换woe后的X

        """

        if not isinstance(fea_woe_dict, dict):
            assert """请传入类似 {'D157': {'0.[-inf, 1.5)': -0.46554351769099783, '1.[1.5, 2.5)': -0.10263802400162944, '2.[2.5, 3.5)': 0.9591358533174893, '3.[3.5, 4.5)': 1.115806812932841, '4.[4.5, 7.5)': 1.1319717497861965, '5.[7.5, inf)': 2.369093204627806, '6.nan': -1.2516811662966312}}"""

        if not fea_woe_dict:
            fea_woe_dict = self.fea_woe_dict

        if getattr(X, 'ndim', 1) == 1:

            if hasattr(X, 'name'):  # pd.Series
                if X.name in fea_woe_dict:
                    fea_name, woe = self._transform(X, fea_woe_dict.get(X.name), fea_name=X.name)
                    return woe
                else:
                    return X

            if len(fea_woe_dict) == 1:
                if DEFAULT_NAME in fea_woe_dict:
                    fea_name, woe = self._transform(X, fea_woe_dict.get(DEFAULT_NAME))
                    return woe
                else:
                    return X

        # X.reset_index(inplace=True)

        _check_duplicated_keys(X)

        ###并行处理
        data_with_bins = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform)(X[col], woe_dict, fea_name=col, **kwargs) for col, woe_dict in fea_woe_dict.items()
            if col in X)
        ###并行处理

        if isinstance(X, dict):
            return dict(data_with_bins)
        else:
            bin_df = pd.DataFrame(dict(data_with_bins), index=X.index)
            X_cols = list(X.columns)
            no_bin_cols = list(set(X_cols) - set(bin_df.columns))
            bin_df[no_bin_cols] = X[no_bin_cols]
            # X.set_index('index', inplace=True)

            # return bin_df.set_index('index')
            return bin_df[X_cols]

    def _transform(self, X, woe_dict={}, fea_name=DEFAULT_NAME, other='min'):
        """

        Args:
            X (DataFrame|array-like): 需要转换woe的X
            woe_dict (dict): 变量和woe值的字典，形如：{0: -0.46554351769099783, 1: -0.10263802400162944}
            fea_name (str): 变量名
            other (str): 未来出现的新值给对应的woe值

        Returns:

        """

        try:
            woe = np.zeros(len(X))
        except:
            woe = np.zeros(np.array(X).size)

        if other == 'min':
            other = np.min(list(woe_dict.values()))
        elif other == 'max':
            other = np.max(list(woe_dict.values()))

        woe[np.isin(X, list(woe_dict.keys()), invert=True)] = other

        for k, v in woe_dict.items():
            woe[X == k] = v

        return fea_name, woe

    def load(self, manual_set_dict):
        """
        自定的woe值
        Args:
            manual_set_dict (dict): map结构的woe值，形如: {'D157': {0: -5.545177444479562, 1: 5.497168225293202}}

        Returns:

        """

        assert isinstance(manual_set_dict, dict), '请传入类似 {\'D157\': {0: -5.545177444479562, 1: 5.497168225293202}}'

        for i in manual_set_dict:
            self.fea_woe_dict[i] = manual_set_dict[i]
        return self

    def export(self, to_dataframe=False, to_json=None, to_csv=None, var_bin_woe={}):
        """
        导出规则到dict或json或csv文件
        Args:
            to_dataframe (bool): 是否导出成pd.DataFrame形式
            to_json (str): 保存成json的路径
            to_csv (str): 保存成csv的路径
            var_bin_woe (dict): {'D157': {0: -5.545177444479562, 1: 5.497168225293202}}

        Returns:
            dict: 分割点规则字典

        """

        if var_bin_woe:
            fea_bin_woe = var_bin_woe
        else:
            fea_bin_woe = copy.deepcopy(self.fea_woe_dict)
            fea_bin_woe = {k: {int(i): j for i, j in v.items()} for k, v in fea_bin_woe.items()}
        if to_json is not None:
            save_json(fea_bin_woe, to_json)

        if to_dataframe or to_csv is not None:
            row = list()
            for var_name in fea_bin_woe:
                for bin, woe in fea_bin_woe[var_name].items():
                    row.append({
                        'feature': var_name,
                        'bins': bin,
                        'woe': woe
                    })

            fea_bin_woe = pd.DataFrame(row)

        if to_csv is not None:
            fea_bin_woe.to_csv(to_csv, index=False)

        return fea_bin_woe
