#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: feature_selection.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-20
'''

import numpy as np
import pandas as pd
import shap
from tqdm import tqdm
from xgboost import XGBClassifier

from .metrics import psi, get_ks
from .statistics import calc_iv
from .transformer import FeatureBin
from .utils import unpack_tuple, select_features_dtypes, is_continuous


def select_features_by_miss(df, nan=None, threshold=0.9, include_cols=[], return_drop=False, only_return_drop=False,
                            if_select_flow=False):
    """
    通过缺失率筛选特征
    Args:
        df (DataFrame): 需要进行特征筛选的数据集
        nan (str, regex, list, dict, Series, int, float, or None): 要替换为空的具体值
        threshold (float): 缺失率筛选阀值
        include_cols (list): 需要进行筛选的特征
        return_drop (bool): 是否返回删除的特征列表
        only_return_drop (bool): 是否只返回删除的特征列表
        if_select_flow (bool): 是否是筛选工作流

    Returns:
        DataFrame: 筛选后的数据集
        list: 删除的特征列表
    """
    if nan is not None:
        df = df.replace(nan, np.nan)

    if include_cols:
        cols = include_cols
    else:
        cols = list(df.columns)

    missing_series = df[cols].isnull().sum() / len(df)

    del_c = list(missing_series[missing_series > threshold].index)

    if if_select_flow:
        return (del_c, threshold,
                pd.DataFrame({'feature': cols, 'miss_rate': missing_series}))  # TODO 需要检查下cols和psi_series是不是一一对应的

    if only_return_drop:
        return del_c

    r = df.drop(columns=del_c)

    res = (r,)
    if return_drop:
        res += (del_c,)

    return unpack_tuple(res)


def select_features_by_concentrate(df, nan=None, threshold=0.9, include_cols=[], return_drop=False,
                                   only_return_drop=False, if_select_flow=False):
    """
    通过集中度筛选特征
    Args:
        df (DataFrame): 需要进行特征筛选的数据集
        nan (str, regex, list, dict, Series, int, float, or None): 要替换为空的具体值
        threshold (float): 集中度筛选阀值
        include_cols (list): 需要进行筛选的特征
        return_drop (bool): 是否返回删除的特征列表
        only_return_drop (bool): 是否只返回删除的特征列表
        if_select_flow (bool): 是否是筛选工作流

    Returns:
        DataFrame: 筛选后的数据集
        list: 删除的特征列表
    """
    if nan is not None:
        df = df.replace(nan, np.nan)

    if include_cols:
        cols = include_cols
    else:
        cols = list(df.columns)
    del_c = []
    max_ratios_ls = []
    row_count = df.shape[0]
    for col in cols:
        max_ratios = max(df[col].value_counts() / row_count)  # 缺失的占比不会被放进来
        # max_ratios = max(df[col].value_counts(dropna=False, normalize=True))    #缺失的占比会被放进来

        max_ratios_ls.append(max_ratios)
        if max_ratios > threshold:
            del_c.append(col)

    if if_select_flow:
        return (del_c, threshold,
                pd.DataFrame(
                    {'feature': cols, 'concentration_rate': max_ratios_ls}))

    if only_return_drop:
        return del_c

    r = df.drop(columns=del_c)

    res = (r,)
    if return_drop:
        res += (del_c,)

    return unpack_tuple(res)


def select_features_by_psi(base, no_base, target='target', threshold=0.05, include_cols=[],
                           return_drop=False, only_return_drop=False, if_select_flow=False, feature_bin=None):
    """
    通过psi筛选特征
    Args:
        base (DataFrame): 基准数据集
        no_base (DataFrame): 非基准数据集
        target (str): 目标变量名称
        threshold (float): psi筛选阀值
        include_cols (list): 需要进行筛选的特征
        return_drop (bool): 是否返回删除的特征列表
        only_return_drop (bool): 是否只返回删除的特征列表
        if_select_flow (bool): 是否是筛选工作流
        feature_bin (autobmt.FeatureBin): 特征分箱对象

    Returns:
        DataFrame: 筛选后的数据集
        list: 删除的特征列表
    """
    if include_cols:
        cols = include_cols
    else:
        cols = list(base.columns)

    if feature_bin is None or feature_bin == False:
        log.info('未进行分箱计算的psi')
        psi_series = psi(no_base[cols], base[cols])
    else:
        if isinstance(feature_bin, FeatureBin):
            for i in cols:  # TODO 待优化，考虑多线程
                if i not in feature_bin.splits_dict:
                    t = base[i]
                    if is_continuous(t):
                        feature_bin.fit(t, base[target], is_need_monotonic=False)
        else:
            feature_bin = FeatureBin()
            to_bin_cols = []
            for i in cols:  # TODO 待优化，考虑多线程
                if is_continuous(base[i]):
                    to_bin_cols.append(i)

            feature_bin.fit(base[to_bin_cols], base[target], is_need_monotonic=False)

        psi_series = psi(feature_bin.transform(no_base[cols]), feature_bin.transform(base[cols]))

    del_c = list(psi_series[psi_series > threshold].index)  # 大于psi阈值的变量删除

    if if_select_flow:
        return (
            del_c, threshold,
            pd.DataFrame({'feature': psi_series.index, 'PSI': psi_series}))

    if only_return_drop:
        return del_c

    r = base.drop(columns=del_c)

    res = (r,)
    if return_drop:
        res += (del_c,)

    return unpack_tuple(res)


def select_features_by_iv(df, target='target', threshold=0.02, include_cols=[], return_drop=False, return_iv=False,
                          only_return_drop=False, if_select_flow=False, feature_bin=None):
    """
    通过iv筛选特征
    Args:
        df (DataFrame): 需要进行特征筛选的数据集
        target (str): 目标变量名称
        threshold (float): iv筛选阀值
        include_cols (list): 需要进行筛选的特征
        return_drop (bool): 是否返回删除的特征列表
        return_iv (bool): 是否返回iv
        only_return_drop (bool): 是否只返回删除的特征列表
        if_select_flow (bool): 是否是筛选工作流
        feature_bin (autobmt.FeatureBin): 特征分箱对象

    Returns:
        DataFrame: 筛选后的数据集
        list: 删除的特征列表
    """
    if include_cols:
        cols = np.array(include_cols)
    else:
        cols = np.array(df.columns)

    iv = np.zeros(len(cols))
    for i in range(len(cols)):
        iv[i] = calc_iv(df[cols[i]], df[target], feature_bin=feature_bin)  # 使用和select_features_by_iv_diff一样的逻辑

    drop_index = np.where(iv < threshold)

    del_c = cols[drop_index]

    if if_select_flow:
        return (del_c, threshold, pd.DataFrame({'feature': cols, 'IV': iv}))

    if only_return_drop:
        return del_c

    r = df.drop(columns=del_c)

    res = (r,)
    if return_drop:
        res += (del_c,)

    if return_iv:
        res += (pd.Series(iv, index=cols),)

    return unpack_tuple(res)


def select_features_by_iv_diff(dev, no_dev, target='target', threshold=2, include_cols=[], return_drop=False,
                               return_iv=False, only_return_drop=False, if_select_flow=False, feature_bin=None):
    """
    通过iv差值筛选特征
    Args:
        dev (DataFrame): 基准数据集
        no_dev (DataFrame): 非基准数据集
        target (str): 目标变量名称
        threshold (float): iv差值筛选阀值
        include_cols (list): 需要进行筛选的特征
        return_drop (bool): 是否返回删除的特征列表
        return_iv (bool): 是否返回iv
        only_return_drop (bool): 是否只返回删除的特征列表
        if_select_flow (bool): 是否是筛选工作流
        feature_bin (autobmt.FeatureBin): 特征分箱对象

    Returns:
        DataFrame: 筛选后的数据集
        list: 删除的特征列表
    """
    if include_cols:
        cols = np.array(include_cols)
    else:
        cols = np.array(dev.columns)

    iv = np.zeros(len(cols))
    iv_no_dev = np.zeros(len(cols))
    if feature_bin is None:
        feature_bin = FeatureBin()
        for i in range(len(cols)):
            iv[i] = calc_iv(dev[cols[i]], dev[target], feature_bin=feature_bin)
            no_dev_feature = feature_bin.transform(no_dev[cols[i]])
            iv_no_dev[i] = calc_iv(no_dev_feature, no_dev[target], feature_bin=feature_bin)
    else:
        for i in range(len(cols)):
            iv[i] = calc_iv(dev[cols[i]], dev[target], feature_bin=feature_bin)
            no_dev_feature = feature_bin.transform(no_dev[cols[i]])
            iv_no_dev[i] = calc_iv(no_dev_feature, no_dev[target], feature_bin=feature_bin)

    # iv_diff = abs(iv - iv_no_dev) * 10
    iv_diff = (iv - iv_no_dev) * 10

    drop_index = np.where(iv_diff > threshold)  # IV差值大于2个点的变量剔除

    del_c = cols[drop_index]

    if if_select_flow:
        return (del_c, threshold,
                pd.DataFrame({'feature': cols, 'dev_nodev_iv_diff': iv_diff}))

    if only_return_drop:
        return del_c

    r = dev.drop(columns=del_c)

    res = (r,)
    if return_drop:
        res += (del_c,)

    if return_iv:
        res += (pd.Series(iv, index=cols),)

    return unpack_tuple(res)


def select_features_by_corr(df, target='target', by='IV', threshold=0.7, include_cols=[], return_drop=False,
                            only_return_drop=False, if_select_flow=False, feature_bin=None):
    """
    通过通过相关性筛选特征
    Args:
        df (DataFrame): 需要进行特征筛选的数据集
        target (str): 目标变量名称
        by (str|array): 用于删除特征的特征权重
        threshold (float): iv筛选阀值
        include_cols (list): 需要进行筛选的特征
        return_drop (bool): 是否返回删除的特征列表
        only_return_drop (bool): 是否只返回删除的特征列表
        if_select_flow (bool): 是否是筛选工作流
        feature_bin (autobmt.FeatureBin): 特征分箱对象

    Returns:
        DataFrame: 筛选后的数据集
        list: 删除的特征列表
    """
    if include_cols:
        cols = include_cols
    else:
        cols = list(df.columns)

    if isinstance(by, pd.DataFrame):
        if by.shape[1] == 1:
            by = pd.Series(by.iloc[:, 0].values, index=by.index)
        else:
            by = pd.Series(by.iloc[:, 1].values, index=by.iloc[:, 0].values)

    if not isinstance(by, (str, pd.Series)):
        by = pd.Series(by, index=df.columns)

    # 计算iv
    if isinstance(by, str):
        df_corr = df[cols].corr().abs()
        # df_corr = df[cols].fillna(-999).corr().abs()
        ix, cn = np.where(np.triu(df_corr.values, 1) > threshold)  # ix是行，cn是列
        if len(ix):
            gt_thre = np.unique(np.concatenate((ix, cn)))
            gt_thre_cols = df_corr.index[gt_thre]
            iv = {}
            for i in gt_thre_cols:
                iv[i] = calc_iv(df[i], target=df[target], feature_bin=feature_bin)

            by = pd.Series(iv, index=gt_thre_cols)

    # 给重要性排下序，倒序
    by = by[list(set(by.index) & set(cols))].sort_values(ascending=False)

    by.index = by.index.astype(type(list(df.columns)[0]))
    df_corr = df[list(by.index)].corr().abs()
    # df_corr = df[list(by.index)].fillna(-999).corr().abs()

    ix, cn = np.where(np.triu(df_corr.values, 1) > threshold)

    del_all = []

    if len(ix):

        for i in df_corr:

            if i not in del_all:
                # 找出与当前特征的相关性大于域值的特征
                del_tmp = list(df_corr[i][(df_corr[i] > threshold) & (df_corr[i] != 1)].index)

                # 比较当前特征与需要删除的特征的特征重要性
                if del_tmp:
                    by_tmp = by.loc[del_tmp]
                    del_l = list(by_tmp[by_tmp <= by.loc[i]].index)
                    del_all.extend(del_l)

    del_c = list(set(del_all))

    if if_select_flow:
        return (del_c, threshold, pd.DataFrame({'feature': df_corr.index}))

    if only_return_drop:
        return del_c

    r = df.drop(columns=del_c)

    res = (r,)
    if return_drop:
        res += (del_c,)

    return unpack_tuple(res)


from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class ShapSelectFeature:
    def __init__(self, estimator, linear=False, estimator_is_fit_final=False):
        self.estimator = estimator
        self.linear = linear
        self.weight = None
        self.estimator_is_fit_final = estimator_is_fit_final

    def fit(self, X, y, exclude=None):
        '''

        Args:
            X:
            y:
            exclude:

        Returns:

        '''
        if exclude is not None:
            X = X.drop(columns=exclude)
        if not self.estimator_is_fit_final:
            self.estimator.fit(X, y)
        if self.linear:
            explainer = shap.LinearExplainer(self.estimator, X)
        else:
            estimator = self.estimator.get_booster()
            temp = estimator.save_raw()[4:]
            estimator.save_raw = lambda: temp
            explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        shap_abs = np.abs(shap_values)
        shap_importance_list = shap_abs.mean(0)
        self.weight = pd.DataFrame(shap_importance_list, index=X.columns, columns=['weight'])
        return self.weight


def feature_select(datasets, fea_names, target, feature_select_method='shap', method_threhold=0.001,
                   corr_threhold=0.8, psi_threhold=0.1, params={}):
    '''

    Args:
        datasets:
        fea_names:
        target:
        feature_select_method:
        method_threhold:
        corr_threhold:
        psi_threhold:
        params:

    Returns:

    '''
    dev_data = datasets['dev']
    nodev_data = datasets['nodev']

    params = {
        'learning_rate': params.get('learning_rate', 0.05),
        'n_estimators': params.get('n_estimators', 200),
        'max_depth': params.get('max_depth', 3),
        #'min_child_weight': params.get('min_child_weight', max(round(len(dev_data) * 0.01), 50)),
        'min_child_weight': params.get('min_child_weight', 5),
        'subsample': params.get('subsample', 0.7),
        'colsample_bytree': params.get('colsample_bytree', 0.9),
        'colsample_bylevel': params.get('colsample_bylevel', 0.7),
        'gamma': params.get('gamma', 7),
        'reg_alpha': params.get('reg_alpha', 10),
        'reg_lambda': params.get('reg_lambda', 10)
    }

    xgb_clf = XGBClassifier(**params)
    xgb_clf.fit(dev_data[fea_names], dev_data[target])

    if feature_select_method == 'shap':
        shap_model = ShapSelectFeature(estimator=xgb_clf, estimator_is_fit_final=True)
        fea_weight = shap_model.fit(dev_data[fea_names], dev_data[target])
        fea_weight.sort_values(by='weight', inplace=True)
        fea_weight = fea_weight[fea_weight['weight'] >= method_threhold]
        log.info('Shap阈值: {}'.format(method_threhold))
        log.info('Shap剔除的变量个数: {}'.format(len(fea_names) - fea_weight.shape[0]))
        log.info('Shap保留的变量个数: {}'.format(fea_weight.shape[0]))
        fea_names = list(fea_weight.index)
        log.info('*' * 50 + 'Shap筛选变量' + '*' * 50)


    elif feature_select_method == 'feature_importance':
        fea_weight = pd.DataFrame(list(xgb_clf.get_booster().get_score(importance_type='gain').items()),
                                  columns=['fea_names', 'weight']
                                  ).sort_values('weight').set_index('fea_names')
        fea_weight = fea_weight[fea_weight['weight'] >= method_threhold]
        log.info('feature_importance阈值: {}'.format(method_threhold))
        log.info('feature_importance剔除的变量个数: {}'.format(len(fea_names) - fea_weight.shape[0]))
        fea_names = list(fea_weight.index)
        log.info('feature_importance保留的变量个数: {}'.format(fea_names))
        log.info('*' * 50 + 'feature_importance筛选变量' + '*' * 50)

    if corr_threhold:
        _, del_fea_list = select_features_by_corr(dev_data[fea_names], by=fea_weight, threshold=corr_threhold,return_drop=True)
        log.info('相关性阈值: {}'.format(corr_threhold))
        log.info('相关性剔除的变量个数: {}'.format(len(del_fea_list)))
        fea_names = [i for i in fea_names if i not in del_fea_list]
        # fea_names = list(set(fea_names) - set(del_fea_list))
        log.info('相关性保留的变量个数: {}'.format(len(fea_names)))
        log.info('*' * 50 + '相关性筛选变量' + '*' * 50)

    if psi_threhold:
        psi_df = psi(dev_data[fea_names], nodev_data[fea_names]).sort_values(0)
        psi_df = psi_df.reset_index()
        psi_df = psi_df.rename(columns={'index': 'fea_names', 0: 'psi'})
        psi_list = psi_df[psi_df.psi < psi_threhold].fea_names.tolist()
        log.info('PSI阈值: {}'.format(psi_threhold))
        log.info('PSI剔除的变量个数: {}'.format(len(fea_names) - len(psi_list)))
        fea_names = [i for i in fea_names if i in psi_list]
        # fea_names = list(set(fea_names) and set(psi_list))
        log.info('PSI保留的变量个数: {}'.format(len(fea_names)))
        log.info('*' * 50 + 'PSI筛选变量' + '*' * 50)

    return fea_names


def stepwise_del_feature(model , datasets, fea_names, target, params={}):
    '''

    Args:
        datasets:
        fea_names:
        target:
        params:

    Returns:

    '''
    log.info("开始逐步删除变量")
    dev_data = datasets['dev']
    nodev_data = datasets['nodev']
    # stepwise_del_params = {
    #     'learning_rate': params.get('learning_rate', 0.05),
    #     'n_estimators': params.get('n_estimators', 200),
    #     'max_depth': params.get('max_depth', 3),
    #     'min_child_weight': params.get('min_child_weight', max(round(len(dev_data) * 0.01), 50)),
    #     'subsample': params.get('subsample', 0.7),
    #     'colsample_bytree': params.get('colsample_bytree', 0.9),
    #     'colsample_bylevel': params.get('colsample_bylevel', 0.7),
    #     'gamma': params.get('gamma', 7),
    #     'reg_alpha': params.get('reg_alpha', 10),
    #     'reg_lambda': params.get('reg_lambda', 10)
    # }

    #xgb_clf = XGBClassifier(**stepwise_del_params)
    model.fit(dev_data[fea_names], dev_data[target])

    pred_test = model.predict_proba(nodev_data[fea_names])[:, 1]
    pred_train = model.predict_proba(dev_data[fea_names])[:, 1]

    test_ks = get_ks(nodev_data[target], pred_test)
    train_ks = get_ks(dev_data[target], pred_train)
    log.info('test_ks is : {}'.format(test_ks))
    log.info('train_ks is : {}'.format(train_ks))

    train_number, oldks, del_list = 0, test_ks, list()
    log.info('train_number: {}, test_ks: {}'.format(train_number, test_ks))

    # while True:
    #     flag = True
    #     for fea_name in tqdm(fea_names):
    #         print('变量{}进行逐步：'.format(fea_name))
    #         names = [fea for fea in fea_names if fea_name != fea]
    #         print('变量names is：', names)
    #         xgb_clf.fit(dev_data[names], dev_data[target])
    #         train_number += 1
    #         pred_test = xgb_clf.predict_proba(nodev_data[names])[:, 1]
    #         test_ks = get_ks(nodev_data[target], pred_test)
    #         if test_ks >= oldks:
    #             oldks = test_ks
    #             flag = False
    #             del_list.append(fea_name)
    #             log.info(
    #                 '等于或优于之前结果 train_number: {}, test_ks: {} by feature: {}'.format(train_number, test_ks, fea_name))
    #             fea_names = names
    #     if flag:
    #         print('=====================又重新逐步==========')
    #         break
    #     log.info("结束逐步删除变量 train_number: %s, test_ks: %s del_list: %s" % (train_number, oldks, del_list))
    #     print('oldks is ：',oldks)
    #     print('fea_names is : ',fea_names)

    for fea_name in tqdm(fea_names):
        names = [fea for fea in fea_names if fea_name != fea]
        model.fit(dev_data[names], dev_data[target])
        train_number += 1
        pred_test = model.predict_proba(nodev_data[names])[:, 1]
        test_ks = get_ks(nodev_data[target], pred_test)
        if test_ks >= oldks:
            oldks = test_ks
            del_list.append(fea_name)
            log.info(
                '等于或优于之前结果 train_number: {}, test_ks: {} by feature: {}'.format(train_number, test_ks, fea_name))
            fea_names = names
    log.info("结束逐步删除变量 train_number: %s, test_ks: %s del_list: %s" % (train_number, oldks, del_list))

    ########################
    log.info('逐步剔除的变量个数: {}'.format(del_list))
    fea_names = [i for i in fea_names if i not in del_list]
    # fea_names = list(set(fea_names) - set(del_list))
    log.info('逐步保留的变量个数: {}'.format(len(fea_names)))
    log.info('*' * 50 + '逐步筛选变量' + '*' * 50)

    return del_list, fea_names


class FeatureSelection:
    def __init__(self, df, target='target', data_type='type',
                 exclude_columns=['key', 'target', 'apply_time', 'type'], params=None,
                 match_dict=None):
        """
        特征选择模块，初始化方法
        Args:
            df (DataFrame): 需要进行变量筛选的数据集
            target (str): 目标值y变量名称
            data_type (str): 数据集划分标示的名称【即划分train、test、oot的字段名称】
            exclude_columns (list): 需要排除的特征
            match_dict (DataFrame): 数据源特征字典
            params (dict): 筛选特征的方法字典，有'empty'、'const'、'psi'、'iv'、'iv_diff'、'corr' 6种筛选方法供选择。字典形如：
            {
                'empty': {'threshold': 0.9},    #若特征的缺失值大于0.9被删除
                'const': {'threshold': 0.95},   #若特征单个值的占比大于0.95被删除
                'psi': {'threshold': 0.05},  #若特征在train、test上psi值大于0.05被删除
                'iv': {'threshold': 0.02},  #若特征的iv值小于0.02被删除
                'iv_diff': {'threshold': 2},    #若特征在train、test上的iv差值乘10后，大于2，特征被删除
                'corr': {'threshold': 0.7}, #若两个特征相关性高于0.7时，iv值低的特征被删除
            }
        """
        self.df = df
        self.target = target
        self.data_type = data_type
        self.exclude_columns = exclude_columns + [self.target]
        self.match_dict = match_dict
        self.params = params
        self.check()
        self.features = [name for name in list(self.df.columns) if name not in self.exclude_columns]
        self.feature_dict = self.get_feature_dict()
        self.select_log_df = self.build_select_log_df()
        self.step_evaluate_log_df = []  # 记录每一步的评估结果

    @property
    def get_features(self):
        """返回当前数据集最新的特征"""
        return [name for name in list(self.df.columns) if name not in self.exclude_columns]

    @property
    def get_evaluate_df_log(self):
        """合并每一步的评估结果"""
        if len(self.step_evaluate_log_df) == 0:
            log.info("并未进行评估过!!!")
            return None
        else:
            evaluate_log_df = pd.concat(self.step_evaluate_log_df, axis=0).reset_index(drop=True)
            return evaluate_log_df

    def get_feature_dict(self):
        """通过数据源简称去数据字典中获取数据源的特征名称"""
        if self.match_dict is not None and isinstance(self.match_dict, pd.DataFrame):
            if self.match_dict.columns.isin(['feature', 'cn']).all():
                model_name_dict_df = pd.DataFrame({'feature': self.features})
                model_name_dict_df = model_name_dict_df.merge(self.match_dict[['feature', 'cn']], on='feature',
                                                              how='left')
                return model_name_dict_df
            else:
                raise KeyError("原始数据字典中没有feature或cn字段，请保证同时有feature字段和cn字段")
        else:
            model_name_dict_df = pd.DataFrame(
                {'feature': self.features, 'cn': ""})
        return model_name_dict_df

    def build_select_log_df(self):
        """返回特征字典，如果需要个性化修改，可以在此方法中修改"""
        return self.feature_dict

    def mapping_selection_func(self):
        """
        特征选择方法映射类
        如果需要增加新的特征选择方法，只需要增加这个字典即可
        注意:python3.6的字典默认会按照顺序进行遍历
        """
        return {
            "empty": select_features_by_miss,
            "const": select_features_by_concentrate,
            "psi": select_features_by_psi,
            "iv": select_features_by_iv,
            "iv_diff": select_features_by_iv_diff,
            "corr": select_features_by_corr,
        }

    def select(self):
        """
        执行定义的特征选择方法，返回筛选过后的数据集，剩余特征名称，以及筛选过程
        Returns:

        """
        log.info('开始执行特征选择模块... 数据集结构为[{}]'.format(self.df.shape))

        if self.params is None:  # 默认只进行3种方式进行变量筛选
            self.params = {
                'empty': {'threshold': 0.9},
                # 'const': {'threshold': 0.95},
                # 'psi': {'threshold': 0.05},
                'iv': {'threshold': 0.02},
                # 'iv_diff': {'threshold': 2},
                'corr': {'threshold': 0.7},
            }
            if self.data_type in self.df:
                self.params['psi'] = {'threshold': 0.05, 'target': self.target}
            log.info('未指定筛选方法的阈值，使用默认方法和阈值：{}'.format(self.params))

        if self.data_type in self.df:
            dev_data = self.df[self.df[self.data_type] == 'train']
            if 'oot' in np.unique(self.df[self.data_type]):
                nodev_data = self.df[self.df[self.data_type] == 'oot']
            else:
                nodev_data = self.df[self.df[self.data_type] == 'test']
        else:
            dev_data = self.df

        fb = FeatureBin()

        for k, v in self.params.items():
            v.update({'if_select_flow': True})
            v.update({'include_cols': self.get_features})
            if k in ['iv', 'iv_diff', 'psi', 'corr']:
                v.update({'feature_bin': fb})
                if 'target' not in v:
                    v.update({'target': self.target})
            # 执行具体的筛选方法
            if k in ['iv_diff', 'psi'] and self.data_type in self.df:
                del_c, th, fea_value_df = self.mapping_selection_func()[k](dev_data, nodev_data, **v)
                log.info('删除变量 ：{}'.format(del_c))
            else:
                del_c, th, fea_value_df = self.mapping_selection_func()[k](dev_data, **v)
                log.info('删除变量 ：{}'.format(del_c))
            if k == 'iv':
                # 将算好的iv值放进去
                self.params['corr']['by'] = fea_value_df

            self.df = self.df.drop(columns=del_c)

            self.select_log_df = self.select_log_df.merge(fea_value_df, on='feature', how='left')
            if k == 'iv':
                step_name = "{}_selection_feature_flag (<{})".format(k, th)
            else:
                step_name = "{}_selection_feature_flag (>{})".format(k, th)
            log.info("{} 方法剔除变量，阈值为{}，剔除的变量有 : {} 个".format(k, step_name, len(del_c)))
            self.select_log_df[step_name] = self.select_log_df['feature'].map(lambda x: 1 if x in del_c else 0)

        # 所有的特征选择方法，只要命中一个，即剔除该特征
        filter_condition_features = [name for name in list(self.select_log_df.columns) if '_feature_flag' in name]
        self.select_log_df['feature_filter_flag'] = self.select_log_df[filter_condition_features].sum(axis=1)
        log.info('特征选择执行完成... 数据集结构为[{}]'.format(self.df.shape))

        # return self.df, self.get_features, self.select_log_df, self.get_evaluate_df_log, fb
        return self.df, self.get_features, self.select_log_df, fb

    def check(self):
        """
        特征选择模块，前置检查，符合要求，则往下运行
        Returns:

        """
        log.info('开始进行前置检查')
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError('数据集不能为空并且数据集必须是dataframe!!!')

        if self.data_type not in self.df:
            log.info('train、test数据集标识的字段名不存在！或未进行数据集的划分，筛选变量无法使用psi、iv_diff进行筛选，建议请将数据集划分为train、test!!!')
            if self.params is not None:
                if 'psi' in self.params:
                    del self.params['psi']
                if 'iv_diff' in self.params:
                    del self.params['iv_diff']
        else:
            data_type_ar = np.unique(self.df[self.data_type])
            if 'train' not in data_type_ar:
                raise KeyError("""没有开发样本，数据集标识字段{}没有`train`该取值!!!""".format(self.data_type))

            if 'test' not in data_type_ar:
                raise KeyError("""没有验证样本，数据集标识字段{}没有`test`该取值!!!""".format(self.data_type))

        if self.target is None:
            raise ValueError('数据集的目标变量名称不能为空!!!')

        if self.target not in self.df:
            raise KeyError('样本中没有目标变量y值!!!')

        if self.exclude_columns is None or self.target not in self.exclude_columns:
            raise ValueError('exclude_columns 不能为空，必须包含target字段!!!')
        n_cols, c_cols, d_cols = select_features_dtypes(self.df, exclude=self.exclude_columns)
        log.info('数值特征个数: {}'.format(len(n_cols)))
        log.info('字符特征个数: {}'.format(len(c_cols)))
        log.info('日期特征个数: {}'.format(len(d_cols)))
        if len(c_cols) > 0:
            log.info('数据集中包含有{}个字符特征,{}个日期特征'.format(len(c_cols), len(d_cols)))
