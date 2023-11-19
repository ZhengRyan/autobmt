#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: utils.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-20
'''

import gc
import json
import math
import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score, r2_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
# from xgboost.sklearn import XGBRegressor
from functools import wraps

import autobmt

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level="info", name=__name__).logger

FILLNA = -999
CONTINUOUS_NUM = 10


def support_dataframe(require_target=True):
    """用于支持dataframe的装饰器
    """

    def sup_df(fn):
        @wraps(fn)
        def func(frame, *args, **kwargs):
            if not isinstance(frame, pd.DataFrame):
                return fn(frame, *args, **kwargs)

            frame = frame.copy()
            if require_target and isinstance(args[0], str):
                target = frame.pop(args[0])
                args = (target,) + args[1:]
            elif 'target' in kwargs and isinstance(kwargs['target'], str):
                kwargs['target'] = frame.pop(kwargs['target'])

            if 'return_bin' not in kwargs:
                kwargs['return_bin'] = True

            res = dict()
            for col in frame:
                r = fn(frame[col], *args, **kwargs)

                if not isinstance(r, np.ndarray):
                    if isinstance(r, tuple):
                        r = r[1]
                    else:
                        r = [r]
                res[col] = r
            return pd.DataFrame(res)

        return func

    return sup_df


def get_splitted_data(df_selected, target, selected_features):
    X = {}
    y = {}

    X['all'] = df_selected[selected_features]
    y['all'] = df_selected[target]

    for name, df in df_selected.groupby('type'):
        X[name] = df[selected_features]
        y[name] = df[target]

    if not X.__contains__('oot'):
        X['oot'] = None
        y['oot'] = None

    return X['all'], y['all'], X['train'], y['train'], X['test'], y['test'], X['oot'], y['oot']


# def to_score(x, A=404.65547022, B=72.1347520444):
#     result = round(A - B * math.log(x / (1 - x)))
#
#     if result < 0:
#         result = 0
#     if result > 1200:
#         result = 1200
#     result = 1200 - result
#     return result

def to_score(x, A=404.65547021957406, B=72.13475204444818, positive_corr=False):
    """
    概率值转分
    Args:
        x (float): 模型预测的概率值
        base_score=600
        odds=15
        pdo=50
        rate=2
        #实际意义为当比率为1/15，输出基准评分600，当比率为基准比率2倍时，1/7.5，基准分下降50分，为550
        A (float): 评分卡offset；；；offset = base_score - (pdo / np.log(rate)) * np.log(odds)
        B (float): 评分卡factor；；；factor = pdo / np.log(rate)
        positive_corr: 分数与模型预测的概率值是否正相关。默认False，负相关，即概率约高，分数越低

    Returns:
        score (float): 标准评分
    """
    result = round(A - B * math.log(x / (1 - x)))

    if positive_corr:
        if result < 0:
            result = 0
        if result > 1200:
            result = 1200
        result = 1200 - result
        return result
    else:
        if result < 300:
            result = 300
        if result > 900:
            result = 900
        return result


def score2p(x, A=404.65547022, B=72.1347520444):
    """
    分转概率
    Args:
        x (float): 标准评分
        pdo=50;rate=2;odds=15;base_score=600
        A (float): 评分卡offset；；；offset = base_score - (pdo / np.log(rate)) * np.log(odds)
        B (float): 评分卡factor；；；factor = pdo / np.log(rate)

    Returns:
        p (float): 概率值
    """
    return 1 / (1 + np.exp((x - A) / B))


def train_test_split_(df_src, target='target', test_size=0.3):
    """
    样本切分函数.先按target分类，每类单独切成train/test，再按train/test合并，
    使得train/test的badrate能高度一致
    Args:
        df_src:
        target:
        test_size:

    Returns:

    """

    l = [[], [], [], []]
    for target_value, X in df_src.groupby(target):

        X[target] = target_value

        row = train_test_split(X.drop(labels=target, axis=1), X[target], test_size=test_size, random_state=1024)

        for i in range(0, 4):
            l[i].append(row[i])

    list_df = []
    for i in range(0, 4):
        list_df.append(pd.concat(l[i]))

    return tuple(list_df)


def split_data_type(df, key_col='id', target='target', apply_time='apply_time', test_size=0.3):
    if df[target].isin([0, 1]).all():
        log.info('样本y值在0，1')
    else:
        log.info('\033[0;31m样本y值不在0，1之间，请检查！！！\033[0m')

    assert df[target].isin([0, 1]).all()

    log.info('样本情况：', df.shape)
    df.drop_duplicates(subset=key_col, inplace=True)
    log.info('分布情况：', df.groupby(target)[key_col].count().sort_index())
    log.info('样本drop_duplicates情况：', df.shape)

    # ---------查看各月badrate---------------------
    df['apply_month'] = df[apply_time].map(lambda s: s[:7])
    log.info(df.groupby('apply_month').describe()[target])
    del df['apply_month']

    # ---------样本划分----------------------------
    ##需要oot
    # df_selected = df_id #can filter records here
    # # df_oot = df_selected[df_selected['apply_time']>= '2019-04-01']
    # # X_train = df_selected[df_selected['apply_time']<= '2019-01-31']
    # # X_test = df_selected[(df_selected['apply_time']> '2019-01-31') & (df_selected['apply_time']< '2019-04-01')]

    # df_oot = df_selected[df_selected['apply_time']>= '2019-03-01']
    # X_train = df_selected[df_selected['apply_time']<= '2018-12-31']
    # X_test = df_selected[(df_selected['apply_time']> '2018-12-31') & (df_selected['apply_time']< '2019-03-01')]

    # #X_train, X_test, y_train, y_test = geo_train_test_split(df_not_oot,label=label)

    # df_id.loc[df_oot.index,'type'] = 'oot'
    ##需要oot

    # 不需要oot的时候运行下面这一行代码
    X_train, X_test, y_train, y_test = train_test_split_(df, target=target, test_size=test_size)
    # X_train, X_test, y_train, y_test = train_test_split(df_id.drop(columns=target), df_id[target], test_size=test_size,
    #                                                     random_state=123)
    # 不需要oot的时候运行下面这一行代码

    df.loc[X_train.index, 'type'] = 'train'
    df.loc[X_test.index, 'type'] = 'test'

    log.info(df.groupby('type').describe()[target])

    # ----------输出---------------------------------
    # df_id.to_csv(data_dir + '{}_split.csv'.format(client_batch), index=False)
    return df


def select_features_dtypes(df, exclude=None):
    """
    根据数据集，筛选出数据类型
    Args:
        df: 数据集
        exclude: 排除不需要参与筛选的列

    Returns:

    """
    if exclude is not None:
        df = df.drop(columns=exclude)
    # 筛选出数值类型列
    numeric_list = df.select_dtypes([np.number]).columns.tolist()

    no_numeric_df = df.select_dtypes(include=['object'])
    # 将object类型的列尝试转成时间类型
    dates_objs_df = no_numeric_df.apply(pd.to_datetime, errors='ignore')
    # 筛选出字符类型列
    objs_list = dates_objs_df.select_dtypes(include=['object']).columns.tolist()
    # 筛选出时间类型列
    # dates_df = list(set(dates_objs_df.columns) - set(objs_df.columns))
    date_list = dates_objs_df.select_dtypes(include=['datetime64']).columns.tolist()

    assert len(numeric_list) + len(objs_list) + len(date_list) == df.shape[1]

    return numeric_list, objs_list, date_list


def filter_miss(df, miss_threshold=0.9):
    """

    Args:
        df (DataFrame): 用于训练模型的数据集
        miss_threshold: 缺失率大于等于该阈值的变量剔除

    Returns:

    """
    names_list = []
    for name, series in df.items():
        n = series.isnull().sum()
        miss_q = n / series.size
        if miss_q < miss_threshold:
            names_list.append(name)
    return names_list


###################
def step_evaluate_models(df, features, target, stepname="", is_turner=False):
    """
    用lr,xgb,评估train/test/oot数据集的auc,ks
    Args:
        df: 数据集，包含y,type,features
        features: 入模特征
        target: 目标值
        stepname: 标识是在哪一步进行评估的
        is_turner: 是否需要进行调参

    Returns:

    """
    X_train = df[df['type'] == 'train'][features]
    y_train = df[df['type'] == 'train'][target]
    data = df[['type', target]]

    # xgb默认参数
    xgb_params = {"base_score": 0.5, "booster": "gbtree", "colsample_bylevel": 1, "colsample_bytree": 0.8, "gamma": 3,
                  "learning_rate": 0.1, "max_delta_step": 0, "max_depth": 6, "min_child_weight": 50,
                  "n_estimators": 200, "n_jobs": -1, "objective": "binary:logistic", "random_state": 0,
                  "reg_alpha": 5, "reg_lambda": 5, "scale_pos_weight": 1,
                  "subsample": 0.8}
    lightgbm_params = {'boosting_type': 'gbdt', 'num_threads': 20,
                       'min_child_weight': 50, 'max_depth': 6,
                       'colsample_bytree': 0.8, 'subsample': 0.8,
                       'num_iterations': 200, 'learning_rate': 0.1, 'verbose': -1
                       }
    rf_params = {'max_depth': 6,
                 'n_estimators': 200,
                 'min_samples_leaf': 60, 'n_jobs': -1, 'min_samples_split': 60,
                 'verbose': 0
                 }
    catboost_params = {'depth': 6, 'l2_leaf_reg': 3,
                       'n_estimators': 200, 'learning_rate': 0.1,
                       'subsample': 0.8
                       }
    if is_turner:
        # 需要调参走这个逻辑
        # models = {
        #     "lr": LogisticRegression().fit(X_train, y_train),
        #     "rf": rf_turner(X_train, y_train),
        #     "xgb": xgb_turner(X_train, y_train, X_test, y_test)[1],
        #     # "lightgbm": lightgbm_turner(X_train, y_train, X_test, y_test)[1],
        #     # "catboost": catboost_turner(X_train, y_train, X_test, y_test)[1]
        # }

        pass
    else:
        # 使用默认参数进行训练评估
        models = {
            "lr": LogisticRegression().fit(X_train, y_train),
            "rf": RandomForestRegressor(**rf_params).fit(X_train, y_train),
            # "xgb": XGBRegressor(**xgb_params).fit(X_train, y_train),
            # "lightgbm": LGBMRegressor(**lightgbm_params).fit(X_train, y_train),
            # "catboost": CatBoostRegressor(**catboost_params, verbose=False).fit(X_train, y_train)
        }

    result = []
    for name, model in models.items():
        # model = model.fit(X_train, y_train)
        if isinstance(model, LogisticRegression):
            data['prob'] = model.predict_proba(df[features])[:, 1]
        elif isinstance(model, RandomForestRegressor):
            data['prob'] = model.predict(df[features])
        # elif isinstance(model, XGBRegressor):
        #     data['prob'] = model.predict(df[features])
        # elif isinstance(model, LGBMRegressor):
        #     data['prob'] = model.predict(df[features])
        # elif isinstance(model, CatBoostRegressor):
        #     data['prob'] = model.predict(df[features])

        df_splitted_type_auc_ks = data.groupby('type').apply(
            lambda df_: pd.Series({'{}_auc'.format(name): autobmt.get_auc(df_[target], df_['prob']),
                                   '{}_ks'.format(name): autobmt.get_ks(df_[target], df_['prob'])}))
        result.append(df_splitted_type_auc_ks)

    evaluate_df = pd.concat(result, axis=1)
    one_step_df = merge_rows_one_row_df(evaluate_df)
    one_step_df['feature_num'] = len(features)  # 加上特征个数
    one_step_df['stepname'] = stepname  # 用于标识是哪一步计算的auc和ks
    xgb_evaluate_log = str(one_step_df.filter(like='xgb').applymap(lambda x: round(x, 4)).iloc[0].to_dict())
    del_df(data)
    del_df(X_train)
    del_df(y_train)
    return one_step_df, xgb_evaluate_log


def model_predict_evaluate(model, df, features, target, A=404.65547022, B=72.1347520444, exclude_cols=None,
                           is_return_var=False):
    """
    1. 利用已经训练好的model对象，对数据集进行预测
    2. 数据集中需要包含有一列type，标识train/test/oot
    3. 返回各个数据集的auc,ks
    4. 返回type,真实y值，预测的概率值，预测的标准分(供后续输入到模型报告中生成外部报告)
    LogisticRegression().fit()
    model = sm.Logit(train_data[target], sm.add_constant(train_data[features])).fit()
    XGBRegressor().fit()
    Args:
        model: 已经训练好的模型对象，支持lr,xgb
        df: 用于训练模型的数据集
        features: 入模变量
        target: 目标值
        A: 评分卡大A
        B: 评分卡大B
        exclude_cols: 样本中的原始字段（非x，y）

    Returns:

    """
    assert 'type' in df.columns.tolist()
    data = df.copy()
    if isinstance(model, LogisticRegression):
        data['p'] = model.pa(data[features])[:, 1]
    elif isinstance(model, statsmodels.discrete.discrete_model.BinaryResultsWrapper):
        log.info("mode is : statsmodels.discrete.discrete_model.BinaryResultsWrapper")
        data['p'] = model.predict(sm.add_constant(data[features]))
    # elif isinstance(model, XGBRegressor):
    #     data['p'] = model.predict(data[features])
    # elif isinstance(model, LGBMRegressor):
    #     data['p'] = model.predict(data[features])
    # elif isinstance(model, CatBoostRegressor):
    #     data['p'] = model.predict(data[features])
    elif isinstance(model, RandomForestRegressor):
        data['p'] = model.predict(data[features])
    data['score'] = data['p'].map(lambda x: to_score(x, A, B))
    evaluate_df = data.groupby('type').apply(
        lambda df: pd.Series({'auc': autobmt.get_auc(df[target], df['p']),
                              'ks': autobmt.get_ks(np.array(df[target]), df['p'])}))

    if is_return_var:
        return evaluate_df, data[exclude_cols + ['p', 'score'] + features]
    else:
        return evaluate_df, data[exclude_cols + ['p', 'score']]


def del_df(df):
    """
    清空一个dataframe
    Args:
        df (DataFrame): 用于训练模型的数据集

    Returns:

    """
    # df.drop(df.index, inplace=True)
    del df
    gc.collect()


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


def get_max_corr_feature(df, features):
    """返回每个变量与其相关性最高的变量以及相关性"""
    corr_df = df.loc[:, features].corr()
    corr_value_series = corr_df.apply(lambda x: x.nlargest(2)[1]).rename("corr_value")
    corr_name_series = corr_df.apply(lambda x: x.nlargest(2).index[1]).rename("corr_name")

    max_corr_df = pd.concat([corr_name_series, corr_value_series], axis=1)
    return max_corr_df


def dump_to_pkl(contents, path):
    pickle.dump(contents, open(path, "wb"))


def load_from_pkl(path):
    return pickle.load(open(path, 'rb'))


def read_sql_string_from_file(path):
    with open(path, 'r', encoding='utf-8') as fb:
        sql = fb.read()
        return sql


def fea_woe_dict_format(fea_woe_dict, splits_dict):
    for j in fea_woe_dict:
        range_format = {int(re.match(r"^(\d+)\.", i).group(1)): i for i in splits_dict[j]}
        fea_woe_dict[j] = {range_format[k]: v for k, v in fea_woe_dict[j].items()}

    return fea_woe_dict


###################

def save_json(res_dict, file, indent=4):
    """
    保存成json文件
    Args:
        res_dict (dict): 需要保存的内容
        file (str|IOBase): 保存后的文件
        indent (int): json文件格式化缩进

    Returns:

    """
    if isinstance(file, str):
        file = open(file, 'w')

    with file as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=indent)


def load_json(file):
    """
    读取json文件
    """
    if isinstance(file, str):
        file = open(file, 'r')

    with file as f:
        res_dict = json.load(f)

    return res_dict


def is_continuous(series):
    series = to_ndarray(series)
    if not np.issubdtype(series.dtype, np.number):
        return False

    n = len(np.unique(series))
    return n > CONTINUOUS_NUM or n / series.size > 0.5
    # return n / series.size > 0.5


def to_ndarray(s, dtype=None):
    """
    """
    if isinstance(s, np.ndarray):
        arr = np.copy(s)
    elif isinstance(s, pd.core.base.PandasObject):
        arr = np.copy(s.values)
    else:
        arr = np.array(s)

    if dtype is not None:
        arr = arr.astype(dtype)
    # covert object type to str
    elif arr.dtype.type is np.object_:
        arr = arr.astype(np.str)

    return arr


def fillna(feature, fillna_va=FILLNA):
    # 复制array 或者 将pandas.core.series.Series变成array
    copy_fea = np.copy(feature)

    mask = pd.isna(copy_fea)

    copy_fea[mask] = fillna_va

    return copy_fea


def split_empty(feature, y=None, return_mask=True):
    copy_fea = np.copy(feature)
    mask = pd.isna(copy_fea)

    copy_fea = copy_fea[~mask]
    if y is not None:
        copy_y = np.copy(y)
        copy_y = copy_y[~mask]
        return copy_fea, copy_y, mask
    return copy_fea, mask


def split_points_to_bin(feature, splits):
    """split points to bin feature
    """
    # log.info("def split_points_to_bin(feature, splits):")
    # log.info(splits)
    feature = fillna(feature)
    return np.digitize(feature, splits)


def np_count(arr, value, default=None):
    c = (arr == value).sum()

    if default is not None and c == 0:
        return default

    return c


def unpack_tuple(x):
    if len(x) == 1:
        return x[0]
    else:
        return x


def split_target(frame, target):
    """
    """
    if isinstance(target, str):
        f = frame.drop(columns=target)
        t = frame[target]
    else:
        f = frame.copy(deep=False)
        t = target

    return f, t


##################g
# corr
def get_corr(df):
    return df.corr()


# accuracy
def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# precision
def get_precision(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return precision_score(y_true, y_pred)
    else:
        return precision_score(y_true, y_pred, average='macro')


# recall
def get_recall(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return recall_score(y_true, y_pred)
    else:
        return precision_score(y_true, y_pred, average='macro')


# f1
def get_f1(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return f1_score(y_true, y_pred)
    else:
        return f1_score(y_true, y_pred, average='macro')


# r2
def r2(preds, target):
    return r2_score(target, preds)


def get_best_threshold(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    ks = list(tpr - fpr)
    thresh = threshold[ks.index(max(ks))]
    return thresh


def get_bad_rate(df):
    return df.sum() / df.count()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


##################g


##################考虑用太极实现
def t_cols_sum_axis_1_np(arr):
    res = np.zeros(arr.shape[0], dtype=float)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            res[i] += arr[i, j]

    return res


def t_cols_sum_axis_0_np(arr):
    res = np.zeros(arr.shape[1], dtype=float)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            res[j] += arr[i, j]

    return res


def t_min_np(arr):
    res = np.inf
    for i in range(arr.shape[0]):
        if res > arr[i]:
            res = arr[i]

    return res


def t_sum_np(arr):
    res = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            res += arr[i, j]

    return res

##################考虑用太极实现
