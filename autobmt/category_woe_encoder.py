#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: category_woe_encoder.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-09-21
'''

import json
import operator
import sys
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def category_2_woe(df, category_cols=[], target='target'):
    """
    每个类别都会转成woe值。缺失值不转，即还是为缺失值。在考虑到未来如果有新类别，给予other对应woe为0
    Args:
        df (DataFrame):
        category_cols (list): 类别变量list
        target:

    Returns:

    """
    var_value_woe = {}
    for i in category_cols:
        bin_g = df.groupby(by=i)[target].agg([('total_cnt', 'count'), ('bad_cnt', 'sum')])
        bin_g['good_cnt'] = bin_g['total_cnt'] - bin_g['bad_cnt']
        bad_count = sum(bin_g['bad_cnt'])
        good_count = sum(bin_g['good_cnt'])
        # bin_g['bad_rate'] = bin_g['bad_cnt'] / sum(bin_g['bad_cnt'])
        bin_g['bad_rate'] = bin_g['bad_cnt'].map(lambda x: 1 / bad_count if x == 0 else x / bad_count)
        # bin_g['good_rate'] = bin_g['good_cnt'] / sum(bin_g['good_cnt'])
        bin_g['good_rate'] = bin_g['good_cnt'].map(lambda x: 1 / good_count if x == 0 else x / good_count)
        # bin_g['good_rate'].replace({0: 0.0000000001}, inplace=True)  # good_rate为0的情况下，woe算出来是-inf。即将0使用一个极小数替换
        # bin_g['woe'] = bin_g.apply(lambda x: 0.0 if x['bad_rate'] == 0 else np.log(x['good_rate'] / x['bad_rate']),
        #                            axis=1)
        bin_g['woe'] = bin_g.apply(lambda x: np.log(x['good_rate'] / x['bad_rate']), axis=1)

        value_woe = bin_g['woe'].to_dict()
        value_woe['other'] = 0  # 未来有新类别的情况下，woe值给予0
        var_value_woe[i] = value_woe

    return var_value_woe


def bin_to_woe(df, var_bin_woe_dict):
    """
    根据传进来的var_bin_woe_dict对原始值进行映射。
    如在var_bin_woe_dict没有的类别（数据集中新出现的类别，归为到other这类）同时var_bin_woe_dict中得有other该类别对应的woe值
    如果var_bin_woe_dict中没有other该类别对应的woe值，即数据集中新出现的类别归为缺失值，即新出现的类别没有woe值
    Args:
        df:
        var_bin_woe_dict (dict):

    Returns:

    """

    for feature, bin_woe in var_bin_woe_dict.items():
        df[feature] = df[feature].map(
            lambda x: x if (x in bin_woe.keys() or x is np.nan or pd.isna(x)) else 'other')
        df[feature] = df[feature].map(bin_woe)

    return df


def data_to_bin(df, bins_dict={}):
    """
    原始数据根据bins_dict进行分箱
    Args:
        df:
        bins_dict: 分箱字典, 形如{'D157': [-999, 1.0, 2.0, 3.0, 5.0, inf]}

    Returns:

    """

    if not isinstance(bins_dict, dict):
        assert '请传入类似 {\'D157\': [-999, 1.0, 2.0, 3.0, 5.0, inf]}'

    data_with_bins = Parallel(n_jobs=-1)(
        delayed(pd.cut)(df[col], bins=bins, right=False, retbins=True) for col, bins in bins_dict.items())
    data_bin = pd.DataFrame([i[0].astype(str) for i in data_with_bins]).T
    b_dict = dict([(i[0].name, i[1].tolist()) for i in data_with_bins])
    if not operator.eq(bins_dict, b_dict):
        assert '传入的分箱和应用后的分箱不对等，请联系开发者'

    return data_bin


def transform(df, var_bin_woe_dict, bins_dict={}):
    """

    Args:
        df:
        var_bin_woe_dict (dict): 形如{"Sex": {"female": -1.5298770033401874, "male": 0.9838327092415774}, "Embarked": {"C": -0.694264203516269, "S": 0.1977338357888416, "other": -0.030202603851420356}}
        bins_dict:

    Returns:
        df (DataFrame): 转换woe后的数据集
    """

    df_ = df.copy()
    if bins_dict:
        df_ = data_to_bin(df, bins_dict=bins_dict)
    return bin_to_woe(df_, var_bin_woe_dict)


def category_2_woe_save(var_value_woe, path=None):
    if path is None:
        path = sys.path[0]

    with open(path + 'category_var_value_woe.json', 'w') as f:
        json.dump(var_value_woe, f)


def category_2_woe_load(path=None):
    with open(path + 'category_var_value_woe.json', 'r') as f:
        var_value_woe = json.load(f)
    return var_value_woe


#########################################测试代码

if __name__ == "__main__":
    from autobmt.utils import select_features_dtypes

    #####读取数据
    to_model_data_path = os.path.join('..','tests','tutorial_data.csv')
    cust_id = 'APP_ID_C'
    target = 'target'  # 目标变量
    data_type = 'type'
    all_data = pd.read_csv(to_model_data_path)
    train_data = all_data[all_data['type'] == 'train']

    n_cols, c_cols, d_cols = select_features_dtypes(train_data, exclude=[cust_id, target, data_type])
    print('数值特征个数: {}'.format(len(n_cols)))
    print('字符特征个数: {}'.format(len(c_cols)))
    print('日期特征个数: {}'.format(len(d_cols)))

    category_2_woe_save_path = os.path.join('..','tests')

    print("类别变量数据处理前", all_data[c_cols])
    if c_cols:
        print('类别变量数据处理')
        # train_data.loc[:, category_cols] = train_data.loc[:, category_cols].fillna('miss')
        # test_data.loc[:, category_cols] = test_data.loc[:, category_cols].fillna('miss')

        var_value_woe = category_2_woe(train_data, c_cols, target=target)
        category_2_woe_save(var_value_woe, '{}'.format(category_2_woe_save_path))
        # var_value_woe = category_2_woe_load('{}'.format(output_dir))
        train_data = transform(train_data, var_value_woe)
        all_data = transform(all_data, var_value_woe)

    print("类别变量数据处理后", all_data[c_cols])
