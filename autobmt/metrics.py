#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: metrics.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-11-05
'''

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .utils import del_df, unpack_tuple


# auc
def get_auc(target, y_pred):
    """
    计算auc值
    Args:
        target (array-like): 目标变量列表
        y_pred (array-like): 模型预测的分数或概率列表

    Returns:
        float: auc值
    """
    if len(np.unique(target)) != 2:
        raise ValueError('the target is not 2 classier target')
    else:
        return roc_auc_score(target, y_pred)


# ks
def get_ks(target, y_pred):
    """
    计算ks值
    Args:
        target (array-like): 目标变量列表
        y_pred (array-like): 模型预测的分数或概率列表

    Returns:
        float: ks值
    """
    df = pd.DataFrame({
        'y_pred': y_pred,
        'target': target,
    })
    crossfreq = pd.crosstab(df['y_pred'], df['target'])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['ks'] = abs(crossdens[0] - crossdens[1])
    ks = max(crossdens['ks'])
    return ks


def get_auc_ks_psi(df, target='target', pred='p'):
    """
    计算auc、ks、psi
    Args:
        df:
        target:
        pred:

    Returns:

    """
    df_pred_dev = df[df['type'] == 'train']
    df_pred_nodev = df[df['type'] == 'test']

    # 计算auc、ks、psi
    test_ks = get_ks(df_pred_nodev[target], df_pred_nodev[pred])
    train_ks = get_ks(df_pred_dev[target], df_pred_dev[pred])
    test_auc = get_auc(df_pred_nodev[target], df_pred_nodev[pred])
    train_auc = get_auc(df_pred_dev[target], df_pred_dev[pred])

    q_cut_list = np.arange(0, 1, 1 / 20)
    bins = np.append(np.unique(np.quantile(df_pred_nodev[pred], q_cut_list)), df_pred_nodev[pred].max() + 0.1)
    df_pred_nodev['range'] = pd.cut(df_pred_nodev[pred], bins=bins, precision=0, right=False).astype(str)
    df_pred_dev['range'] = pd.cut(df_pred_dev[pred], bins=bins, precision=0, right=False).astype(str)
    nodev_psi = psi(df_pred_nodev['range'], df_pred_dev['range'])
    res_dict = {'dev_auc': train_auc, 'nodev_auc': test_auc, 'dev_ks': train_ks, 'nodev_ks': test_ks,
                'nodev_dev_psi': nodev_psi}
    del_df(df_pred_dev)
    del_df(df_pred_nodev)
    return pd.DataFrame([res_dict])


def psi(no_base, base, return_frame=False, featurebin=None):
    """
    计算psi值
    Args:
        no_base (DataFrame|array-like): 非基准数据集
        base (DataFrame|array-like):基准数据集
        return_frame (bool): 是否需要返回占比

    Returns:
        float|Series:psi值
    """

    if featurebin is not None:
        if isinstance(featurebin, (dict, list)):
            from .transformer import FeatureBin
            featurebin = FeatureBin().manual_bin(featurebin)

            no_base = featurebin.transform(no_base, labels=True)
        base = featurebin.transform(base, labels=True)

    psi = list()
    frame = list()

    if isinstance(no_base, pd.DataFrame):
        for col in no_base:
            p, f = calc_psi(no_base[col], base[col])
            psi.append(p)
            frame.append(f)

        psi = pd.Series(psi, index=no_base.columns, name='psi')

        frame = pd.concat(
            frame,
            keys=no_base.columns,
            names=['columns', 'id'],
        ).reset_index()
        frame = frame.drop(columns='id')
    else:
        psi, frame = calc_psi(no_base, base)

    res = (psi,)

    if return_frame:
        res += (frame,)

    return unpack_tuple(res)


def calc_psi(no_base, base):
    """
    psi计算的具体逻辑
    Args:
        no_base (array-like): 非基准数据集
        base (array-like): 基准数据集

    Returns:
        float,DataFrame : psi值，占比
    """
    no_base_prop = pd.Series(no_base).value_counts(normalize=True, dropna=False)
    base_prop = pd.Series(base).value_counts(normalize=True, dropna=False)

    psi = np.sum((no_base_prop - base_prop) * np.log(no_base_prop / base_prop))

    frame = pd.DataFrame({
        'no_base': no_base_prop,
        'base': base_prop,
    })
    frame.index.name = 'value'

    return psi, frame.reset_index()


def SSE(y_pred, y):
    """sse
    """
    return np.sum((y_pred - y) ** 2)


def MSE(y_pred, y):
    """mse
    """
    return np.mean((y_pred - y) ** 2)


def AIC(y_pred, y, k, llf=None):
    """AIC信息准则

    Args:
        y_pred (array-like)
        y (array-like)
        k (int): x的数量
        llf (float): 对数似然函数的值
    """
    if llf is None:
        llf = np.log(SSE(y_pred, y))

    return 2 * k - 2 * llf


def BIC(y_pred, y, k, llf=None):
    """贝叶斯信息准则

    Args:
        y_pred (array-like)
        y (array-like)
        k (int): x的数量
        llf (float): 对数似然函数的值
    """
    n = len(y)
    if llf is None:
        llf = np.log(SSE(y_pred, y))

    return np.log(n) * k - 2 * llf


def get_metrics_info(df, by=['apply_mon'], feature_type='td', target='target', data_type='type',
                     apply_time='apply_time'):
    if data_type not in df.columns:
        data_type = 'None'
        df[data_type] = 'None'
    auc_ks = df.groupby(by).apply(
        lambda df_tmp: pd.Series({
            'auc': get_auc(df_tmp[target], df_tmp[feature_type]),
            'ks': get_ks(df_tmp[target], df_tmp[feature_type]),
            '正样本': sum(df_tmp[target]),
            '总数': len(df_tmp),
            '正样本占比': np.mean(df_tmp[target]),
            'apply_time': f"{df_tmp[apply_time].min()}至{df_tmp[apply_time].max()}",
            'data_type': f"{list(set(df_tmp[data_type]))}",
        })
    )
    all_auc_ks = pd.DataFrame([{'all': 'all', 'auc': get_auc(df[target], df[feature_type]),
                                'ks': get_ks(df[target], df[feature_type]), '正样本': sum(df[target]), '总数': len(df),
                                '正样本占比': np.mean(df[target]),
                                'apply_time': f"{df[apply_time].min()}至{df[apply_time].max()}",
                                'data_type': f"{list(set(df[data_type]))}", }]).set_index('all')
    all_auc_ks.index.name = auc_ks.index.name
    res = auc_ks.append(all_auc_ks)
    res.index.name = feature_type
    return res


def psi_by_col(df, by_col='apply_mon'):
    by_col_v = sorted(list(set(df[by_col])))
    by_col_psi_lis = []
    for n, j in enumerate(by_col_v):
        by_col_d = df[df[by_col] == j]
        ###计算PSI
        by_col_psi = psi(by_col_d[['score']], df[['score']], )
        by_col_psi.name = f"{j}_PSI"
        by_col_psi_lis.append(by_col_psi)
    by_col_psi_df = pd.DataFrame(by_col_psi_lis).T
    by_col_psi_df['MaxPSI'] = by_col_psi_df.max(axis=1)
    return by_col_psi_df
