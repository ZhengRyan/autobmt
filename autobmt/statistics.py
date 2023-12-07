#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: statistics.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-03
'''

import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

import autobmt
from .metrics import psi, get_auc, get_ks
from .utils import is_continuous, to_ndarray, np_count, support_dataframe, split_points_to_bin


# TODO compare_inflection_point和calc_iv_and_inflection_point考虑要合并

def compare_inflection_point(df_summary):
    """
    比较数据集的拐点趋势
    切记，数据集使用type区分，因此一定要包含type列，数据集是分好箱的数据集，不是原始数据集而是转woe后的
    因为需要转换计算badrate
    Args:
        df_summary (DataFrame):分好箱的数据集，可能包含train/test/oot

    Returns:
        DataFrame: 拐点信息
    """

    def __calc_group(data):
        range = data.range.tolist()
        nan_bin = '{}.nan'.format(len(range) - 1)
        if nan_bin in range:
            data = data[~data['range'].isin([nan_bin])]
        badrate = data.positive_rate
        is_monotonic = badrate.is_monotonic_decreasing or badrate.is_monotonic_increasing

        inflection_point_index, inflection_shape = get_inflection_point_index(badrate)
        return pd.Series({'is_monotonic': is_monotonic, 'bin_count': len(range),
                          'inflection_point_num': len(inflection_point_index),
                          'inflection_point_index': inflection_point_index,
                          'inflection_shape': inflection_shape})

    data_inflection_df = df_summary.groupby('var_name').apply(__calc_group)
    return data_inflection_df


def merge_rows_one_row_df(df, name="", stepname=None):
    """将一个多行的dataframe合并成只有一行的dataframe"""
    tmp_arr = []
    for i in range(df.shape[0]):
        tmp = df.iloc[i, :].add_prefix("{}_{}".format(df.index[i], name))
        tmp_arr.append(tmp)

    result_df = pd.DataFrame(pd.concat(tmp_arr, axis=0)).T
    if stepname is not None:  # 合并成一行后,增加一列标识，用于和别的评估进行区分
        result_df['stepname'] = stepname
    return result_df


def calc_iv_and_inflection_point(df, target='target', bin_func=None, bin_format={}):
    """
    计算iv，拐点
    Args:
        df (DataFrame):分好箱转换后的数据集
        target (str):目标变量
        bin_func (str): 分箱方法
        bin_format (dict): 格式化好的分箱点

    Returns:
        DataFrame
    """

    def __calc_group(data, bin_func):
        range = data.range.tolist()
        nan_bin = '{}.nan'.format(len(range) - 1)
        if nan_bin in range:
            data = data[~data['range'].isin([nan_bin])]
        bin_badrate = data.positive_rate
        inflection_point_index, inflection_shape = get_inflection_point_index(bin_badrate)
        return pd.Series({'IV': data.IV.get(0), 'bin_count': len(range),
                          'inflection_point_num': len(inflection_point_index),
                          'inflection_point_index': inflection_point_index,
                          'inflection_shape': inflection_shape, 'bin_func': bin_func})

    summary = Parallel(n_jobs=-1)(
        delayed(calc_bin_summary)(df[[col, target]], bin_col=col, bin=False, target=target, is_sort=False,
                                  bin_format=bin_format) for col in df.columns if col not in [target])
    var_summary = pd.concat(summary, axis=0)

    var_iv_inflection_df = var_summary.groupby('var_name').apply(__calc_group, bin_func)
    return var_iv_inflection_df


def calc_var_summary(df, bin_format={}, include_cols=[], target='target', need_bin=True, **kwargs):
    """
    计算所有变量的详情
    Args:
        df (DataFrame): 含有目标变量及分箱后的数据集
        bin_format (dict): 格式化好的分箱点
        include_cols (list): 需要统计的特征
        target (str): 目标值变量名称
        need_bin (bool): 是否需要分箱

    Returns:
        DataFrame: 变量分箱详情
    """

    if include_cols:
        cols = np.array(include_cols)
    else:
        cols = np.array(df.columns)

    if not need_bin:
        kwargs = {'bin': False}
    summary = Parallel(n_jobs=-1)(
        delayed(calc_bin_summary)(df[[col, target]], bin_col=col, target=target, is_sort=False, bin_format=bin_format,
                                  **kwargs) for col in cols if col != target)
    var_summary = pd.concat(summary, axis=0)
    return var_summary


def calc_bin_summary(df, bin_col='score', bin=10, target='target', is_sort=True, method='equal_freq',
                     is_need_monotonic=False, bin_format={}, **kwargs):
    """
    变量分箱详情
    Args:
        df (DataFrame): 含目标变量的数据集
        bin_col (str): 需要计算的列名
        bin (int): 分几箱
        target (str):目标变量列名
        method (str):分箱方法；'dt'、'chi'、'equal_freq'、'kmeans'四种供选择
        is_sort (bool):是否需要排序，默认True，倒序排序
        **kwargs:

    Returns:
        DataFrame: 变量分箱详情
    """

    # negative:good，positive:bad
    def __calc_group(data_, var_name):
        """获取分组的人数,好坏人数"""
        count = len(data_)
        bad_num = data_.y.sum()
        good_num = count - bad_num
        return pd.Series(
            {'var_name': var_name, 'min': min(data_.x), 'max': max(data_.x), 'positive': bad_num, 'negative': good_num,
             'total': count})

    data = pd.DataFrame({'x': df[bin_col], 'y': df[target]})
    total = len(data)
    positive_count = data.y.sum()
    negative_count = total - positive_count

    data['range'] = 0
    if bin is False:
        data['range'] = data.x
    elif isinstance(bin, (list, np.ndarray, pd.Series)):
        if len(bin) < len(data.x):
            bin = split_points_to_bin(data.x, bin)

        data['range'] = bin
    elif isinstance(bin, int):
        fb = autobmt.FeatureBin()
        fb.fit(data.x, data.y, n_bins=bin, method=method, is_need_monotonic=is_need_monotonic, **kwargs)
        data['range'] = fb.transform(data.x)
        bin_format = fb.export()
        bin_format[bin_col] = bin_format.pop('x')

    bin_g = data.groupby(data['range'], dropna=False).apply(__calc_group, var_name=bin_col)

    if is_sort:
        bin_g.sort_values(by='min', ascending=False, inplace=True)  # 正常

    bin_g['positive_rate'] = bin_g['positive'] / bin_g['total']  # bad_rate,,,区间_正样本比率
    bin_g['negative_rate'] = bin_g['negative'] / bin_g['total']  # 区间_负样本比率

    bin_g['odds'] = bin_g['positive'] / bin_g['negative']

    # bin_g['positive_pct'] = bin_g['positive'] / positive_count  # 区间正样本占比
    bin_g['positive_pct'] = bin_g['positive'].map(
        lambda x: 1 / positive_count if x == 0 else x / positive_count)  # 区间正样本占比
    # bin_g['negative_pct'] = bin_g['negative'] / negative_count  # 区间负样本/总体负样本
    bin_g['negative_pct'] = bin_g['negative'].map(
        lambda x: 1 / negative_count if x == 0 else x / negative_count)  # 区间负样本/总体负样本
    bin_g['total_pct'] = bin_g['total'] / total  # 区间总人数/总体总人数

    bin_g['cum_negative_pct'] = bin_g['negative_pct'].cumsum()  # 累计负样本人数占比

    cum_positive = bin_g['positive'].cumsum()  # 累计正样本人数
    cum_total = bin_g['total'].cumsum()  # 累计总人数

    bin_g['cum_total_pct'] = cum_total / total  # 累计通过人数占比，累计捕获率，agg2['cum_total_prop'] = cum_total / all_total
    bin_g['cum_positive_pct'] = bin_g['positive_pct'].cumsum()  # 查全率,,,累计正样本人数占比
    bin_g['cum_positive_rate'] = cum_positive / cum_total  # 查准率,,,累计捕获的样本中正样本的占比

    bin_g['ks'] = bin_g['cum_positive_pct'] - bin_g['cum_negative_pct']  # 累计正样本人数占比/累计负样本人数占比

    bin_g['lift'] = bin_g['positive_rate'] / (positive_count / total)
    if bin_g['ks'].sum() < 0:
        bin_g['ks'] = -bin_g['ks']
        bin_g['cum_negative_pct'] = bin_g.loc[::-1, 'negative_pct'].cumsum()[::-1]  # 累计负样本人数占比
        cum_positive_rev = bin_g.loc[::-1, 'positive'].cumsum()[::-1]  # 累计正样本人数
        cum_total_rev = bin_g.loc[::-1, 'total'].cumsum()[::-1]  # 累计总人数
        bin_g['cum_total_pct'] = cum_total_rev / total  # 累计通过人数占比，累计捕获率，agg2['cum_total_prop'] = cum_total / all_total
        bin_g['cum_positive_pct'] = bin_g.loc[::-1, 'positive_pct'].cumsum()[::-1]  # 查全率,,,累计正样本人数占比
        bin_g['cum_positive_rate'] = cum_positive_rev / cum_total_rev  # 查准率,,,累计捕获的样本中正样本的占比

    bin_g['cum_lift'] = bin_g['cum_positive_pct'] / bin_g['cum_total_pct']

    bin_g['woe'] = bin_g.apply(lambda x: np.log(x['positive_pct'] / x['negative_pct']), axis=1)
    bin_g['iv'] = (bin_g['positive_pct'] - bin_g['negative_pct']) * bin_g.woe
    bin_g['IV'] = bin_g.iv.sum()

    bin_g.index.name = 'range'
    bin_g = bin_g.reset_index()

    if bin_col in bin_format:
        range_format = {int(re.match(r"^(\d+)\.", i).group(1)): i for i in bin_format[bin_col]}
        bin_g['range_num'] = bin_g['range']
        bin_g['range'] = bin_g['range'].map(range_format)
    else:
        bin_g['range_num'] = bin_g['range'].fillna(len(bin_g))

    return bin_g


def calc_woe_iv(df, col_name='default_name', bin_format={}, target='target'):
    """
    已对齐IV的计算方式
    计算单变量详情，woe,iv值
    Args:
        df: 含有目标变量及分箱后的数据集
        col_name: 单变量名称
        bin_format: 格式化好的分箱点
        target: 目标值变量名称

    Returns:

    """

    def __calc_group(data_, var_name):
        """获取分组的人数,好坏人数"""
        count = len(data_)
        bad_num = data_.Y.sum()
        good_num = count - bad_num

        return pd.Series({'var_name': var_name, 'Total': count, 'Bad': bad_num, 'Good': good_num})

    X, Y = df[col_name], df[target]

    data = pd.DataFrame({'X': X, 'Y': Y})

    bin_g = data.groupby(data['X'], dropna=False).apply(__calc_group, var_name=col_name)
    total = data.Y.count()
    bad_count = (data.Y == 1).sum()
    good_count = (data.Y == 0).sum()
    bin_g['Bad_Rate'] = bin_g['Bad'] / bin_g['Total']  # bad_rate
    # bin_g['Pct_Bad'] = bin_g['Bad'] / bad_count  # bad_人数占比
    # bin_g['Pct_Good'] = bin_g['Good'] / good_count  # good_人数占比
    bin_g['Pct_Bad'] = bin_g['Bad'].map(lambda x: 1 / bad_count if x == 0 else x / bad_count)  # bad_人数占比
    bin_g['Pct_Good'] = bin_g['Good'].map(lambda x: 1 / good_count if x == 0 else x / good_count)  # good_人数占比
    bin_g['Pct_Bin'] = bin_g['Total'] / total  # 总人数占比
    # bin_g['累计坏人数'] = bin_g['Good'].cumsum()
    # bin_g['累计好人数'] = bin_g['Good'].cumsum()
    # bin_g['累计坏人数占比'] = bin_g['Good'].cumsum() / bad_count
    # bin_g['累计好人数占比'] = bin_g['Good'].cumsum() / good_count
    # bin_g['woe'] = bin_g.apply(
    #     lambda x: 0.0 if x['Pct_Good'] == 0 or x['Pct_Bad'] == 0 else round(np.log(x['Pct_Bad'] / x['Pct_Good']),
    #                                                                         5),
    #     axis=1)
    bin_g['woe'] = bin_g.apply(lambda x: np.log(x['Pct_Bad'] / x['Pct_Good']), axis=1)
    # bin_g['ks'] = abs(bin_g['累计坏人数占比'] - bin_g['累计好人数占比'])
    bin_g['iv'] = (bin_g['Pct_Bad'] - bin_g['Pct_Good']) * bin_g.woe
    bin_g['IV'] = bin_g.iv.replace({np.inf: 0, -np.inf: 0}).sum()
    bin_g.index.name = 'range'
    bin_g = bin_g.reset_index()
    if col_name in bin_format:
        range_format = {int(re.match(r"^(\d+)\.", i).group(1)): i for i in bin_format[col_name]}
        bin_g['range_num'] = bin_g['range']
        bin_g['range'] = bin_g['range'].map(range_format)
    else:
        bin_g['range_num'] = bin_g['range'].fillna(len(bin_g))

    # 每个分箱字段之间加上一个空行
    # bin_g = bin_g.append(pd.Series(),ignore_index=True)
    return bin_g


def get_inflection_point_index(arr):
    """
    返回一个数组的拐点索引,以及单调情况
    Args:
        arr (array) : 数组

        + 单调递增
        - 单调递减
        u u形曲线
        ~u 倒u形曲线
        ~ 不单调
    Returns:
        array: 拐点的位置
        str: 单调标志
    """
    diff_arr = np.diff(arr).tolist()
    # 返回是单调递增(+)，还是单调递减(-)，还是不单调(~)
    monotonic_flag = check_monotonic(diff_arr)
    index_arr = []  # 记录拐点的位置
    for i in range(0, len(diff_arr) - 1):
        if np.signbit(diff_arr[i]) != np.signbit(diff_arr[i + 1]):
            index_arr.append(i + 1)
    if len(index_arr) == 1:
        monotonic_flag = "~U" if arr[1] - arr[0] > 0 else "U"

    return index_arr, monotonic_flag


def check_monotonic(arr):
    """判断是单调递增还是单调递减,先调用arr = np.diff(arr)"""
    count = np.sum(np.array(arr) > 0)
    if count == len(arr):
        return "+"  # 单调递增
    elif count == 0:
        return "-"  # 单调递减
    else:
        return "~"  # 不单调


def get_vif(frame):
    """
    计算VIF
    Args:
        frame:  (ndarray|DataFrame)

    Returns:
        Series
    """
    index = None
    if isinstance(frame, pd.DataFrame):
        index = frame.columns
        frame = frame.values

    l = frame.shape[1]
    vif = np.zeros(l)

    for i in range(l):
        X = frame[:, np.arange(l) != i]
        y = frame[:, i]

        model = sm.OLS(y, X)
        r2 = model.fit().rsquared_adj

        vif[i] = 1 / (1 - r2)

    return pd.Series(vif, index=index, name='vif')


def WOE(y_prob, n_prob):
    """计算woe值

    Args:
        y_prob: 正样本在整个正样本中的占比
        n_prob: 负样本在整个负样本中的占比

    Returns:
        number: woe 值
    """
    return np.log(y_prob / n_prob)


def _IV(feature, target):
    """计算IV值

    Args:
        feature (array-like)
        target (array-like)

    Returns:
        number: IV值
    """
    feature = to_ndarray(feature, dtype='str')
    target = to_ndarray(target)

    value = 0

    for v in np.unique(feature):
        y_prob, n_prob = probability(target, mask=(feature == v))
        value += (y_prob - n_prob) * WOE(y_prob, n_prob)

    return value


@support_dataframe()
def calc_iv(feature, target, feature_bin=None, **kwargs):
    """计算1个特征的IV值

    Args:
        feature (array-like)
        target (array-like)
        n_bins (int): 需要分几箱
        method (str): 分箱方法；'dt'、'chi'、'equal_freq'、'kmeans'四种供选择， 默认 'dt'
        **kwargs (): bin_method_run分箱函数的其它参数
    """

    ###TODO: 考虑增加是否单调的参数

    if is_continuous(feature):

        if feature_bin is not None:
            if hasattr(feature, 'name') and feature.name in feature_bin.splits_dict:
                feature = feature_bin.transform(feature)
            else:
                feature = feature_bin.fit_transform(feature, target, method='dt', is_need_monotonic=False, **kwargs)
        else:
            if 'return_bin' in kwargs: del kwargs['return_bin']
            s, feature = autobmt.bin_method_run(feature, target, return_bin=True, is_need_monotonic=False, **kwargs)

    return _IV(feature, target)


def bin_badrate(feature, target=None):
    """
    计算badrate【即正样本占比】
    Args:
        feature:
        target:

    Returns:

    """
    badrate_list = []
    bin_rate_list = []
    total = len(feature)
    uni_fea = np.sort(np.unique(feature))
    # uni_fea = np.unique(feature)
    if target is None:
        for value in uni_fea:
            # mask = (feature == value)
            mask = feature == value
            bin_rate_list.append(np.sum(mask) / total)
        return min(bin_rate_list)
    else:
        for value in uni_fea:
            # mask = (feature == value)
            mask = feature == value
            bin_target = target[mask]
            bin_badrate = np.sum(bin_target) / len(bin_target)
            bin_rate_list.append(len(bin_target) / total)
            # bin_rate_list.append(np.sum(mask) / total)
            badrate_list.append(bin_badrate)
        return badrate_list, min(bin_rate_list)


def bin_badratebase(feature, target):
    badrate_list = []
    bin_rate_list = []
    total = len(feature)
    uni_fea = np.sort(np.unique(feature))
    # uni_fea = np.unique(feature)
    for value in uni_fea:
        # mask = (feature == value)
        mask = feature == value
        bin_target = target[mask]
        bin_badrate = np.sum(bin_target) / len(bin_target)
        bin_rate_list.append(len(bin_target) / total)
        badrate_list.append(bin_badrate)

    return badrate_list, min(bin_rate_list)


def probability(target, mask=None):
    """计算目标变量占比
    """
    if mask is None:
        return 1, 1

    counts_0 = np_count(target, 0, default=1)
    counts_1 = np_count(target, 1, default=1)

    sub_target = target[mask]

    sub_0 = np_count(sub_target, 0, default=1)
    sub_1 = np_count(sub_target, 1, default=1)

    y_prob = sub_1 / counts_1
    n_prob = sub_0 / counts_0

    return y_prob, n_prob


def get_iv_psi(df, feature_list=[], target='target', by_col='apply_mon', only_psi=True):
    """
    计算iv、psi
    Args:
        df (DataFrame): 原始数据集
        feature_list (list): 需要计算iv、psi的变量
        target (str): 目标变量名称
        by_col (str): 根据哪个变量分组计算iv、psi
        only_psi (bool): 是否只计算psi

    Returns:
        DataFrame: 变量的psi、iv
    """
    fb = autobmt.FeatureBin()
    fb.fit(df[feature_list], df[target], method='equal_freq', is_need_monotonic=False)
    # fb.fit(df[feature_list], df[target], method='dt')
    dev = fb.transform(df)
    by_col_v = sorted(list(set(df[by_col])))

    month_IV = pd.DataFrame()
    month_PSI_lis = []
    for n, j in enumerate(by_col_v):
        by_col_d = dev[dev[by_col] == j]
        if not only_psi:
            ###计算IV
            iv = {}
            iv_all = {}
            for i in feature_list:
                iv[i] = calc_iv(by_col_d[i], target=by_col_d[target], feature_bin=fb)
                if n == 0:
                    iv_all[i] = calc_iv(dev[i], target=dev[target], feature_bin=fb)
            if iv_all:
                month_IV = pd.concat([month_IV, pd.DataFrame([iv_all]).T.rename(columns={0: 'IV'})], axis=1)
            month_IV = pd.concat([month_IV, pd.DataFrame([iv]).T.rename(columns={0: f"{j}_IV"})], axis=1)
        ###计算PSI
        by_col_psi = psi(by_col_d[feature_list], dev[feature_list])
        by_col_psi.name = f"{j}_PSI"
        month_PSI_lis.append(by_col_psi)
    month_PSI = pd.DataFrame(month_PSI_lis).T
    month_PSI['MaxPSI'] = month_PSI.max(axis=1)

    ###iv趋势
    if not month_IV.empty:
        s_col_n = [f"{i}_IV" for i in by_col_v]
        for i in feature_list:
            month_IV.loc[i, 'IV趋势'] = get_inflection_point_index(month_IV.loc[i, s_col_n])[1]

    res = pd.concat([month_IV, month_PSI], axis=1)
    return res.sort_values(by='MaxPSI', ascending=False, )


from IPython.display import display
from pandas._libs import lib
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.groupby import Grouper
from pandas.core.indexes.api import Index
import warnings

is_scalar = lib.is_scalar

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)  # 设置行数为无限制
pd.set_option('display.max_columns', None)  # 设置列数为无限制


def _convert_by(by):
    if by is None:
        by = []
    elif (
        is_scalar(by)
        or isinstance(by, (np.ndarray, Index, ABCSeries, Grouper))
        or hasattr(by, "__call__")
    ):
        by = [by]
    else:
        by = list(by)
    return by


# # 正负样本在不同数据集不同数据集_不同设备的匹配率===一体

def get_pivot_table(df, index_col_name=None, columns_name=None, target='target'):
    index_col_name = _convert_by(index_col_name)
    if target not in df:
        raise KeyError('数据中目标变量y值名称错误！！！')
    df['dfid'] = 1

    zb = pd.pivot_table(df, values='dfid', index=index_col_name, columns=[target], aggfunc='count',
                        margins=True,
                        margins_name='合计')
    zb['正样本占比'] = zb[1] / zb['合计']
    # sb_os = zb

    concat_lis = []
    if columns_name:
        for n, i in enumerate(columns_name):
            sb = pd.pivot_table(df, values='dfid', index=index_col_name, columns=[i], aggfunc='count',
                                margins=True,
                                margins_name='合计')
            sb.drop(columns='合计', inplace=True)
            concat_lis.append(sb)

    concat_lis.append(zb)
    sb_os = pd.concat(concat_lis, axis=1)
    return sb_os


def get_pivot_table_posnegasam(df, index_col_name=None, columns_name=None, target='target'):
    index_col_name = _convert_by(index_col_name)
    columns_name = _convert_by(columns_name)
    if target not in df:
        raise KeyError('数据中目标变量y值名称错误！！！')
    df['dfid'] = 1
    concat_lis = []
    for n, i in enumerate(columns_name):
        sb = pd.pivot_table(df, values='dfid', index=index_col_name,
                            columns=[i, target],
                            aggfunc='count', margins=True, margins_name='合计')

        if n != len(columns_name) - 1:
            sb.drop(columns='合计', inplace=True)

        concat_lis.append(sb)

        zb = pd.pivot_table(df, values=target, index=index_col_name, columns=[i],
                            aggfunc=['mean'],
                            margins=True, margins_name='合计')

        if n != len(columns_name) - 1:
            zb.drop(columns=('mean', '合计'), inplace=True)

        zb.columns = zb.columns.swaplevel(0, 1)
        zb.columns = pd.MultiIndex.from_frame(pd.DataFrame(zb.columns.tolist()).replace({'mean': '正样本占比'}))

        concat_lis.append(zb)

    sb_os = pd.concat(concat_lis, axis=1)
    sb_os.columns = pd.MultiIndex.from_frame(pd.DataFrame(list(sb_os.columns)).replace({'': '样本个数'}))
    sb_os.columns.names = ['', '']
    return sb_os


def get_mr(original_df, match_df, index_col_name=None, columns_name=None, target='target'):
    sb_os = get_pivot_table(match_df, index_col_name, columns_name, target)

    ##############分割线
    ys_sb_os = get_pivot_table(original_df, index_col_name, columns_name, target)

    ###匹配率
    pp_ratio = sb_os / ys_sb_os
    del pp_ratio['正样本占比']

    index = [tuple(['匹配样本数', i]) for i in list(sb_os.columns)]
    sb_os.columns = pd.MultiIndex.from_tuples(index)

    index = [tuple(['原始样本数', i]) for i in list(ys_sb_os.columns)]
    ys_sb_os.columns = pd.MultiIndex.from_tuples(index)

    index = [tuple(['匹配率', i]) for i in list(pp_ratio.columns)]
    pp_ratio.columns = pd.MultiIndex.from_tuples(index)
    # pp_ratio.columns = pp_ratio.columns.map(lambda x: str(x) + '_匹配率')

    tmp = pd.merge(ys_sb_os, sb_os, how='left', left_index=True, right_index=True, suffixes=('_原始', '_匹配'))

    res = pd.merge(pp_ratio, tmp, how='left', left_index=True, right_index=True)

    return res


def get_posnegasam_mr(original_df, match_df, index_col_name=None, columns_name=None, target='target'):
    if not columns_name:
        return pd.DataFrame()
        # raise KeyError('columns_name参数是空的！！！调用get_mr函数即可')

    sb_os = get_pivot_table_posnegasam(match_df, index_col_name, columns_name, target)

    #################分割线

    ys_sb_os = get_pivot_table_posnegasam(original_df, index_col_name, columns_name, target)

    ###匹配率
    pp_ratio = sb_os / ys_sb_os
    pp_ratio.drop('正样本占比', axis=1, level=1, inplace=True)

    index = [tuple(['匹配样本数'] + list(i)) for i in list(sb_os.columns)]
    sb_os.columns = pd.MultiIndex.from_tuples(index)

    index = [tuple(['原始样本数'] + list(i)) for i in list(ys_sb_os.columns)]
    ys_sb_os.columns = pd.MultiIndex.from_tuples(index)

    index = [tuple(['匹配率'] + list(i)) for i in list(pp_ratio.columns)]
    pp_ratio.columns = pd.MultiIndex.from_tuples(index)
    pp_ratio.columns = pd.MultiIndex.from_frame(pd.DataFrame(list(pp_ratio.columns)).replace({'样本个数': '总匹配率'}))
    pp_ratio.columns.names = ['', '', '']

    tmp = pd.merge(ys_sb_os, sb_os, how='left', left_index=True, right_index=True)
    res = pd.merge(pp_ratio, tmp, how='left', left_index=True, right_index=True)
    return res


def get_model_auc_ks(match_df, index_col_name=None, columns_name=None, target='target', pred='p'):
    groupby_list = _convert_by(index_col_name)
    columns_name = _convert_by(columns_name)
    if target not in match_df:
        raise KeyError('数据中目标变量y值名称错误！！！')
    if pred not in match_df:
        raise KeyError('数据中模型预测的概率值名称错误！！！')

    auc_lis = []
    ks_lis = []
    if columns_name:
        for i in columns_name:
            # auc
            try:
                type_na_device_type_auc = match_df[match_df[target].notnull()].groupby(
                    groupby_list + [i]).apply(
                    lambda tmp: pd.Series(
                        {'auc': get_auc(tmp[target], tmp[pred])}))
            except:
                type_na_device_type_auc = match_df[match_df[target].notnull()].groupby(
                    groupby_list + [i]).apply(
                    lambda tmp: pd.Series({'auc': 0}))
            auc_lis.append(type_na_device_type_auc.unstack())

            # ks
            try:
                type_na_device_type_ks = match_df[match_df[target].notnull()].groupby(
                    groupby_list + [i]).apply(
                    lambda tmp: pd.Series({'ks': get_ks(tmp[target], tmp[pred])}))
            except:
                type_na_device_type_ks = match_df[match_df[target].notnull()].groupby(
                    groupby_list + [i]).apply(
                    lambda tmp: pd.Series({'ks': 0}))

            ks_lis.append(type_na_device_type_ks.unstack())

    ###auc
    try:
        type_na_auc = match_df[match_df[target].notnull()].groupby(groupby_list).apply(
            lambda tmp: pd.Series({'auc': get_auc(tmp[target], tmp[pred])}))
        index = pd.MultiIndex.from_tuples([('auc', 'all')])
    except:
        type_na_auc = match_df[match_df[target].notnull()].groupby(groupby_list).apply(
            lambda tmp: pd.Series({'auc': 0}))
        index = pd.MultiIndex.from_tuples([('auc', 'all')])
    type_na_auc.columns = index

    auc_lis.append(type_na_auc)

    datatype_didtype_auc = pd.concat(auc_lis, axis=1)

    ###ks
    try:
        type_na_ks = match_df[match_df[target].notnull()].groupby(groupby_list).apply(
            lambda tmp: pd.Series({'ks': get_ks(tmp[target], tmp[pred])}))
        index = pd.MultiIndex.from_tuples([('ks', 'all')])
    except:
        type_na_ks = match_df[match_df[target].notnull()].groupby(groupby_list).apply(
            lambda tmp: pd.Series({'ks': 0}))
        index = pd.MultiIndex.from_tuples([('ks', 'all')])
    type_na_ks.columns = index

    ks_lis.append(type_na_ks)

    datatype_didtype_ks = pd.concat(ks_lis, axis=1)

    datatype_didtype_auc_ks = pd.concat([datatype_didtype_auc, datatype_didtype_ks], axis=1)
    return datatype_didtype_auc_ks


class StatisticsMrAucKs:

    def __init__(self, conf_dict={}, data_path=''):
        self.cust_id = conf_dict.get('cust_id', 'device_id')  # 主键
        self.target_na = conf_dict.get('target_na', 'target')  # 目标变量列名
        self.device_type_na = conf_dict.get('device_type_na', 'device_type')  # 设备列名
        self.year_month_na = conf_dict.get('year_month_na', 'apply_month')
        self.date_na = conf_dict.get('date_na', 'apply_time')  # 时间列名
        self.type_na = conf_dict.get('type_na', 'type')  # 数据集列名
        self.model_pred_res = conf_dict.get('model_pred_res', 'p')  # 模型预测值

        ###设备号列名
        self.oaid_col_na = conf_dict.get('oaid_col_na', 'oaid_md5')
        self.imei_col_na = conf_dict.get('imei_col_na', 'imei_md5')
        self.idfa_col_na = conf_dict.get('idfa_col_na', 'idfa_md5')

        ###设备号的具体取值
        self.oaid_value = conf_dict.get('oaid_value', 'oaid')
        self.imei_value = conf_dict.get('imei_value', 'imei')
        self.idfa_value = conf_dict.get('idfa_value', 'idfa')

        self.data_path = data_path

    def statistics_model_mr_auc_ks(self, by_cols=['device_type', 'os', 'media']):
        model_res = pd.read_csv(self.data_path)

        model_res[self.year_month_na] = model_res[self.date_na].map(lambda x: x[:7])

        model_res[self.device_type_na] = model_res[self.device_type_na].replace(
            {0: self.oaid_value, 1: self.imei_value, 2: self.idfa_value})


        p_null = model_res[model_res[self.model_pred_res].isnull()]

        if len(p_null[p_null[self.device_type_na].isnull()]) > 0:

            print("device_type有为空！！！")
            mdn_oaid = p_null[p_null[self.oaid_col_na].notnull()].rename(columns={self.oaid_col_na: 'device_id'})
            mdn_imei = p_null[p_null[self.imei_col_na].notnull()].rename(columns={self.imei_col_na: 'device_id'})
            mdn_idfa = p_null[p_null[self.idfa_col_na].notnull()].rename(columns={self.idfa_col_na: 'device_id'})

            mdn_oaid[self.device_type_na] = 0  # 0,oaid
            mdn_imei[self.device_type_na] = 1  # 1,imei
            mdn_idfa[self.device_type_na] = 2  # 2,idfa
            p_null_did = mdn_oaid.append(mdn_imei).append(mdn_idfa)

            p_null_did = p_null_did.sort_values([self.device_type_na]).drop_duplicates([self.cust_id],
                                                                                       keep='first')  # first是取0(oaid)、1(imei)、2(idfa)
            del p_null_did['device_id']

            p_null_did[self.device_type_na] = p_null_did[self.device_type_na].replace(
                {0: self.oaid_value, 1: self.imei_value, 2: self.idfa_value})

            model_res = model_res[model_res[self.model_pred_res].notnull()].append(p_null_did)

        model_res['os'] = model_res[self.device_type_na].replace(
            {self.oaid_value: 'android', self.imei_value: 'android', self.idfa_value: 'ios'})

        print('总：', model_res.shape)
        print('没有p值的：', p_null.shape)
        p_notnull = model_res[model_res[self.model_pred_res].notnull()]
        print('有p值的：', p_notnull.shape)

        file_name = self.data_path.split('.csv')[0] + '_模型mr_auc_ks_res——new.xlsx'

        writer = pd.ExcelWriter(file_name)

        ###不同数据集_不同设备的匹配率

        res = get_mr(model_res, p_notnull, self.type_na, by_cols)
        print('1.1、不同集匹配率')
        display(res)

        start_row = 1
        res.to_excel(writer, sheet_name='1、匹配详情', startrow=start_row)  # 指定从哪一行开始写入数据，默认值为0
        # 修改下一次开始写入数据的行位置
        start_row = start_row + res.shape[0] + 4

        ###不同月份_不同设备的匹配率

        res = get_mr(model_res, p_notnull, self.year_month_na, by_cols)
        print('1.2、不同月份匹配率')
        display(res)

        res.to_excel(writer, sheet_name='1、匹配详情', startrow=start_row)
        start_row = start_row + res.shape[0] + 5

        ###正负样本不同数据集_不同设备的匹配率

        res = get_posnegasam_mr(model_res, p_notnull, self.type_na, by_cols)
        print('1.3、不同集、不同设备匹配率')
        display(res)

        res.to_excel(writer, sheet_name='1、匹配详情', startrow=start_row)
        start_row = start_row + res.shape[0] + 5

        res = get_posnegasam_mr(model_res, p_notnull, self.year_month_na, by_cols)
        print('1.4、不同月份、不同设备匹配率')
        display(res)

        res.to_excel(writer, sheet_name='1、匹配详情', startrow=start_row)

        ### 效果统计

        ### 不同集

        res = get_model_auc_ks(p_notnull, self.type_na, by_cols)
        print('2.1、不同集效果')
        display(res)

        start_row = 1
        res.to_excel(writer, sheet_name='2、模型效果详情', startrow=start_row)
        start_row = start_row + res.shape[0] + 4

        ### 不同月份

        res = get_model_auc_ks(p_notnull, self.year_month_na, by_cols)
        print('2.2、不同月份效果')
        display(res)

        res.to_excel(writer, sheet_name='2、模型效果详情', startrow=start_row)
        start_row = start_row + res.shape[0] + 4

        ### 不同集上不同月份

        res = get_model_auc_ks(p_notnull, [self.type_na, self.year_month_na], by_cols)
        print('2.3、不同集、月份效果')
        display(res)

        res.to_excel(writer, sheet_name='2、模型效果详情', startrow=start_row)

        # # 不同集上不同月份

        # # 效果统计结束

        writer.save()
        writer.close()
