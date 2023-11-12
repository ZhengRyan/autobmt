#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: feature_binning.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-20
'''

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, _tree

from .statistics import calc_iv_and_inflection_point, bin_badrate
from .utils import fillna, split_empty, split_points_to_bin, t_sum_np, t_min_np, \
    t_cols_sum_axis_0_np, t_cols_sum_axis_1_np, support_dataframe

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level="info", name=__name__).logger

DEFAULT_BINS = 10
IS_EMPTY_BIN_DEFAULT_BINS = 9


def equal_freq_bin(feature, target=None, min_sample_rate=0.05,
                   n_bins=None,
                   q_cut_list=None, is_need_monotonic=True, is_empty_bin=True):
    """
    等频分箱

    Args:
        feature (array) : 某个x特征
        target (array) : 目标值y变量
        min_sample_rate (number) : 每个箱子的最小占比
        n_bins (int) : 需要分成几箱
        q_cut_list (array、list) : 百分比分割点列表
        is_need_monotonic (bool) : 是否强制单调
        is_empty_bin (bool) : 是否将空箱单独归为一个箱子

    Returns:
        array (array): 分割点
    """

    empty_mask = np.array([])
    if is_empty_bin:
        if target is None:
            feature, empty_mask = split_empty(feature, target)
        else:
            feature, target, empty_mask = split_empty(feature, target)
    else:
        feature = fillna(feature)

    if n_bins is None and q_cut_list is None and is_empty_bin and empty_mask.any():
        n_bins = IS_EMPTY_BIN_DEFAULT_BINS
    elif n_bins is None and q_cut_list is None:
        n_bins = DEFAULT_BINS

    if q_cut_list is None:
        q_cut_list = np.arange(0, 1, 1 / n_bins)

    is_monotonic = False
    while not is_monotonic:
        splits_tmp = np.quantile(feature, q_cut_list)
        splits = np.unique(splits_tmp)[1:]

        x_bins = split_points_to_bin(feature, splits)  # 返回0、1、2、3这样的数值分箱
        if target is None:
            _ = bin_badrate(x_bins)
            is_monotonic = True if _ >= min_sample_rate else False  # 判断最小分箱占比是否满足
        else:
            bin_badrate_li, _ = bin_badrate(x_bins, target)
            is_monotonic = True if _ >= min_sample_rate else False  # 判断最小分箱占比是否满足
            if is_need_monotonic:
                if is_monotonic:  # 满足最小分箱占比后，判断是否满足单调性
                    is_monotonic = pd.Series(bin_badrate_li).is_monotonic_decreasing or pd.Series(
                        bin_badrate_li).is_monotonic_increasing

        if n_bins <= 2:
            break

        n_bins = q_cut_list.size - 1  # 这种是从初始的n_bins(10)，一点一点的减
        q_cut_list = np.arange(0, 1, 1 / n_bins)

    if is_empty_bin and empty_mask.any():
        splits = np.append(splits, np.nan)

    return splits


def dt_bin(feature, target, min_sample_rate=0.05, n_bins=None,
           is_need_monotonic=True, is_empty_bin=True, **kwargs):
    """
    决策树分箱

    Args:
        feature (array) : 某个x特征
        target (array) : 目标值y变量
        min_sample_rate (number) : 每个箱子的最小占比
        n_bins (int) : 需要分成几箱
        is_need_monotonic (bool) : 是否强制单调
        is_empty_bin (bool) : 是否将空箱单独归为一个箱子
        **kwargs : 决策树的其它参数

    Returns:
        array (array): 分割点
    """

    empty_mask = np.array([])
    if is_empty_bin:
        feature, target, empty_mask = split_empty(feature, target)
    else:
        feature = fillna(feature)

    if n_bins is None and is_empty_bin and empty_mask.any():
        n_bins = IS_EMPTY_BIN_DEFAULT_BINS
    elif n_bins is None:
        n_bins = DEFAULT_BINS

    is_monotonic = False
    while not is_monotonic:

        # 决策树分箱逻辑
        '''
        1、初始n_b：10（point），splits：9（point）
        2、9+3=12（point） 》11（bin）
        3、11（bin）- 1 ---》10（bin）
        4、老版本有，新版本跳过
        5、10 - 1 = 9（point）
        '''
        tree = DecisionTreeClassifier(
            min_samples_leaf=min_sample_rate,
            max_leaf_nodes=n_bins,
            # 优先满足min_samples_leaf参数。在满足min_samples_leaf参数参数后，再考虑max_leaf_nodes。
            # 比如情况1：min_samples_leaf设置成0.05，max_leaf_nodes设置成20。满足0.05后，最大max_leaf_nodes只有10，那也就这样了
            # 比如情况2：min_samples_leaf设置成0.05，max_leaf_nodes设置成6。满足0.05后，最大max_leaf_nodes有10，那再考虑max_leaf_nodes，继续分到满足max_leaf_nodes=6停止
            # ps:min_samples_leaf=1表示没有限制
            **kwargs,
        )
        tree.fit(feature.reshape((-1, 1)), target)
        thresholds = tree.tree_.threshold
        thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]
        splits = np.sort(thresholds)
        # 决策树分箱逻辑

        is_monotonic = True
        if is_need_monotonic:
            x_bins = split_points_to_bin(feature, splits)  # 返回0、1、2、3这样的数值分箱

            bin_badrate_li, _ = bin_badrate(x_bins, target)

            # 不需要判断，tree里面已经判断了
            # is_monotonic = True if _ >= min_sample_rate else False  # 判断最小分箱占比是否满足

            is_monotonic = pd.Series(bin_badrate_li).is_monotonic_decreasing or pd.Series(
                bin_badrate_li).is_monotonic_increasing

        n_bins = len(splits) + 1  # 初始n_bins为10，对应的splits为9。需要+1，在后面的n_bins -= 1后，n_bins才会是9

        if n_bins <= 2:
            break
        n_bins -= 1

    if is_empty_bin and empty_mask.any():
        splits = np.append(splits, np.nan)
    return splits


def kmeans_bin(feature, target=None, min_sample_rate=0.05,
               n_bins=None,
               random_state=1, is_need_monotonic=True, is_empty_bin=True):
    """
    kmeans聚类分箱

    Args:
        feature (array) : 某个x特征
        target (array) :  目标值y变量
        min_sample_rate (number) : 每个箱子的最小占比
        n_bins (int): 需要分成几箱
        random_state (int): kmeans模型中的随机种子
        is_need_monotonic (bool) : 是否强制单调
        is_empty_bin (bool) : 是否将空箱单独归为一个箱子

    Returns:
        array (array): 分割点
    """

    empty_mask = np.array([])
    if is_empty_bin:
        if target is None:
            feature, empty_mask = split_empty(feature, target)
        else:
            feature, target, empty_mask = split_empty(feature, target)
    else:
        feature = fillna(feature)

    if n_bins is None and is_empty_bin and empty_mask.any():
        n_bins = IS_EMPTY_BIN_DEFAULT_BINS
    elif n_bins is None:
        n_bins = DEFAULT_BINS

    is_monotonic = False
    while not is_monotonic:

        # kmeans 逻辑
        kmeans = KMeans(
            n_clusters=n_bins,
            random_state=random_state
        )
        kmeans.fit(feature.reshape((-1, 1)), target)

        centers = np.sort(kmeans.cluster_centers_.reshape(-1))

        l = len(centers) - 1
        splits = np.zeros(l)
        for i in range(l):
            splits[i] = (centers[i] + centers[i + 1]) / 2
        # kmeans 逻辑

        x_bins = split_points_to_bin(feature, splits)  # 返回0、1、2、3这样的数值分箱
        if target is None:
            _ = bin_badrate(x_bins)
            is_monotonic = True if _ >= min_sample_rate else False  # 判断最小分箱占比是否满足
        else:
            bin_badrate_li, _ = bin_badrate(x_bins, target)
            is_monotonic = True if _ >= min_sample_rate else False  # 判断最小分箱占比是否满足
            if is_need_monotonic:
                if is_monotonic:  # 满足最小分箱占比后，判断是否满足单调性
                    is_monotonic = pd.Series(bin_badrate_li).is_monotonic_decreasing or pd.Series(
                        bin_badrate_li).is_monotonic_increasing

        n_bins = len(splits) + 1  # 初始n_bins为10，对应的splits为9。需要+1，在后面的n_bins -= 1后，n_bins才会是9

        if n_bins <= 2:
            break
        n_bins -= 1

    if is_empty_bin and empty_mask.any():
        splits = np.append(splits, np.nan)

    return splits


def chi_bin(feature, target, balance=True, min_sample_rate=0.05, n_bins=None,
            is_need_monotonic=True, is_empty_bin=True, min_threshold=None):
    """
    卡方分箱

    Args:
        feature (array) : 某个x特征
        target (array) :  目标值y变量
        min_sample_rate (number) : 每个箱子的最小占比
        n_bins (int): 需要分成几箱
        is_need_monotonic (bool) : 是否强制单调
        is_empty_bin (bool) : 是否将空箱单独归为一个箱子
        min_threshold (number): 最小的卡方阀值

    Returns:
        array (array): 分割点
    """

    empty_mask = np.array([])
    if is_empty_bin:
        feature, target, empty_mask = split_empty(feature, target)
    else:
        feature = fillna(feature)

    if n_bins is None and min_threshold is None and is_empty_bin and empty_mask.any():
        n_bins = IS_EMPTY_BIN_DEFAULT_BINS
    elif n_bins is None and min_threshold is None:
        n_bins = DEFAULT_BINS

    if min_sample_rate and min_sample_rate < 1:
        min_sample_rate = len(feature) * min_sample_rate

    ###
    target_unique = np.unique(target)
    feature_unique = np.unique(feature)
    len_f = len(feature_unique)
    len_t = len(target_unique)

    grouped = np.zeros((len_f, len_t), dtype=float)

    for r in range(len_f):
        tmp = target[feature == feature_unique[r]]
        for c in range(len_t):
            grouped[r, c] = (tmp == target_unique[c]).sum()

    is_monotonic = False
    while not is_monotonic:

        # 卡方 逻辑
        while True:  # 内循环

            ###bmt
            # 判断卡方分箱是否同时满足最小分箱占比和箱子个数
            if len(grouped) <= n_bins and t_min_np(t_cols_sum_axis_1_np(grouped)) >= min_sample_rate:
                break
            ###bmt

            # 计算每一组的卡方
            l = len(grouped) - 1
            chi_list = np.zeros(l, dtype=float)
            chi_min = np.inf
            # chi_ix = []
            for i in range(l):
                chi = 0
                couple = grouped[i:i + 2, :]
                total = t_sum_np(couple)
                cols = t_cols_sum_axis_0_np(couple)
                # t_cols_sum_axis_1(couple)
                # rows = x
                rows = t_cols_sum_axis_1_np(couple)

                for j in range(couple.shape[0]):
                    for k in range(couple.shape[1]):
                        e = rows[j] * cols[k] / total
                        if e != 0:
                            chi += (couple[j, k] - e) ** 2 / e

                # 平衡卡方值
                if balance:
                    chi *= total

                chi_list[i] = chi

                if chi == chi_min:
                    chi_ix.append(i)
                    continue

                if chi < chi_min:
                    chi_min = chi
                    chi_ix = [i]

                # if chi < chi_min:
                #     chi_min = chi

            # 当最小值大于阈值时中断循环
            if min_threshold and chi_min > min_threshold:
                break

            # 获取最小卡方值的那组索引
            min_ix = np.array(chi_ix)
            # min_ix = np.where(chi_list == chi_min)[0]

            # 获取需要删除的索引
            drop_ix = min_ix + 1

            # 按索引合并
            retain_ix = min_ix[0]
            last_ix = retain_ix
            for ix in min_ix:
                # set a new group
                if ix - last_ix > 1:
                    retain_ix = ix

                # 将所有连续索引合并为一组
                for p in range(grouped.shape[1]):
                    grouped[retain_ix, p] = grouped[retain_ix, p] + grouped[ix + 1, p]

                last_ix = ix

            # 删除分组
            grouped = np.delete(grouped, drop_ix, axis=0)
            feature_unique = np.delete(feature_unique, drop_ix)

        # 卡方 逻辑

        splits = feature_unique[1:]

        is_monotonic = True
        if is_need_monotonic:
            x_bins = split_points_to_bin(feature, splits)  # 返回0、1、2、3这样的数值分箱
            bin_badrate_li, _ = bin_badrate(x_bins, target)

            # 不需要判断，内循环里面已经判断了
            # is_monotonic = True if _ >= min_sample_rate else False  # 判断最小分箱占比是否满足

            is_monotonic = pd.Series(bin_badrate_li).is_monotonic_decreasing or pd.Series(
                bin_badrate_li).is_monotonic_increasing

        n_bins = len(splits) + 1  # 初始n_bins为10，对应的splits为9。需要+1，在后面的n_bins -= 1后，n_bins才会是9
        if n_bins <= 2:
            break
        n_bins -= 1

    if is_empty_bin and empty_mask.any():
        splits = np.append(splits, np.nan)

    return splits


@support_dataframe(require_target=False)
def bin_method_run(feature, target=None, method='dt', return_bin=False, **kwargs):
    """
    对数据进行分箱
    Args:
        feature (array-like) : 某个x特征
        target (array-like) :  目标值y变量
        method (str): 分箱方法；'dt'、'chi'、'equal_freq'、'kmeans'四种供选择
        return_bin (bool): 是否返回分箱后的数据
        min_sample_rate (number) : 每个箱子的最小占比
        n_bins (int): 需要分成几箱
        is_need_monotonic (bool) : 是否强制单调
        is_empty_bin (bool) : 是否将空箱单独归为一个箱子
        min_threshold (number): 最小的卡方阀值

    Returns:
        array: 分割点
        array: 原始数据用分箱点替换后的数据

    """

    if method == 'dt':
        splits = dt_bin(feature, target, **kwargs)
    elif method == 'chi':
        splits = chi_bin(feature, target, **kwargs)
    elif method == 'equal_freq':
        splits = equal_freq_bin(feature, target, **kwargs)
    elif method == 'kmeans':
        splits = kmeans_bin(feature, target, **kwargs)
    else:
        splits = np.array([])

    ##返回splits
    if return_bin:
        bins = np.zeros(len(feature))
        if np.isnan(splits[-1]):
            mask = pd.isna(feature)
            bins[~mask] = split_points_to_bin(feature[~mask], splits[:-1])
            bins[mask] = len(splits)
        else:
            bins = split_points_to_bin(feature, splits)

        # return splits, pd.Series(bins, name=feature.name)
        return splits, bins

    return splits


def best_binning(df, x_list=[], target='target', **kwargs):
    """
    最优分箱
    Args:
        df (DataFrame) : 需要分箱的数据集
        x_list (list): 需要分箱的特征列表
        target (str): 目标变量
        **kwargs: 'dt'、'chi'、'equal_freq'、'kmeans'四种分箱方法的分箱参数

    Returns:

    """
    from .transformer import FeatureBin

    assert df[target].isin([0, 1]).all(), 'ERROR: :-) {} :-) 目标变量不是0/1值，请检查！！！'.format(target)
    iv_inflection_arr = []
    cutoff_dic = {}
    # for method in ['equal_freq', 'chi', 'dt', 'kmeans']:
    for method in ['equal_freq', 'chi', 'dt']:
        log.info("正在执行最优分箱之 [{}] ".format(method))

        fb = FeatureBin()
        fb.fit(df[x_list], df[target], method=method, **kwargs)
        cutoff_dic[method] = fb.splits_dict
        var_iv_inflection_df = calc_iv_and_inflection_point(fb.transform(df, labels=True)[x_list + [target]],
                                                            target=target, bin_func=method)
        iv_inflection_arr.append(var_iv_inflection_df.reset_index())

        log.info("执行最优分箱之 [{}] over!!!".format(method))

    # 分析获取最优分箱的结果
    iv_inflection_df = pd.concat(iv_inflection_arr, axis=0)
    best_binning_result = iv_inflection_df.groupby('var_name').apply(
        lambda x: x.sort_values(['inflection_point_num', 'IV', 'bin_count'], ascending=[True, False, False]).head(
            1)).set_index("var_name")

    # 找到各个变量最优分箱的分箱方法
    best_binning_func_index = best_binning_result['bin_func'].to_dict()
    best_cutoff = {k: cutoff_dic[v][k] for k, v in best_binning_func_index.items()}
    fb.manual_bin(best_cutoff)

    # 最优分箱，多返回一个数据集记录选择最优的过程
    best_binning_result = best_binning_result.reset_index('var_name')
    return fb, best_binning_result
