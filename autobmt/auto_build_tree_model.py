#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: auto_build_tree_model.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-11-17
'''

import gc
import json
import os
import time
import warnings

import joblib
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb

import autobmt

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class AutoBuildTreeModel():
    def __init__(self, datasets, fea_names, target, key='key', data_type='type',
                 no_feature_names=['key', 'target', 'apply_time', 'type'], ml_res_save_path='model_result',
                 AB={}, positive_corr=False):

        if data_type not in datasets:
            raise KeyError('train、test数据集标识的字段名不存在！或未进行数据集的划分，请将数据集划分为train、test！！！')

        data_type_ar = np.unique(datasets[data_type])
        if 'train' not in data_type_ar:
            raise KeyError("""没有开发样本，数据集标识字段{}没有`train`该取值！！！""".format(data_type))

        if 'test' not in data_type_ar:
            raise KeyError("""没有验证样本，数据集标识字段{}没有`test`该取值！！！""".format(data_type))

        if target not in datasets:
            raise KeyError('样本中没有目标变量y值！！！')

        # fea_names = [i for i in fea_names if i != key and i != target]
        fea_names = [i for i in fea_names if i not in no_feature_names]
        log.info('数据集变量个数 : {}'.format(len(fea_names)))
        log.info('fea_names is : {}'.format(fea_names))

        self.datasets = datasets
        self.fea_names = fea_names
        self.target = target
        self.key = key
        self.no_feature_names = no_feature_names
        self.ml_res_save_path = os.path.join(ml_res_save_path, time.strftime('%Y%m%d%H%M%S_%S', time.localtime()))
        self.AB = AB
        self.positive_corr = positive_corr  # 分数与模型预测的概率值是否正相关。默认False，负相关，即概率约高，分数越低
        self.min_child_samples = max(round(len(datasets[datasets['type'] == 'train']) * 0.02),
                                     50)  # 一个叶子上数据的最小数量. 可以用来处理过拟合
        # self.min_child_samples = max(round(len(datasets['dev']) * 0.02), 50)  # 一个叶子上数据的最小数量. 可以用来处理过拟合

        os.makedirs(self.ml_res_save_path, exist_ok=True)

    def fit(self, is_feature_select=True, is_auto_tune_params=True, is_stepwise_del_feature=True,
            feature_select_method='shap', method_threhold=0.001,
            corr_threhold=0.8, psi_threhold=0.2):
        '''

        Args:
            is_feature_select:
            is_auto_tune_params:
            feature_select_method:
            method_threhold:
            corr_threhold:
            psi_threhold:

        Returns: xgboost.sklearn.XGBClassifier或lightgbm.sklearn.LGBClassifier；list
            返回最优模型，入模变量list

        '''
        log.info('*' * 30 + '开始自动建模' + '*' * 30)

        log.info('*' * 30 + '获取变量名和数据集' + '*' * 30)
        fea_names = self.fea_names.copy()
        dev_data = self.datasets[self.datasets['type'] == 'train']
        nodev_data = self.datasets[self.datasets['type'] == 'test']

        del self.datasets;
        gc.collect()

        # dev_data = self.datasets['dev']
        # nodev_data = self.datasets['nodev']

        # params = {
        #     'learning_rate': 0.05,
        #     'n_estimators': 200,
        #     'max_depth': 3,
        #     'min_child_weight': 5,
        #     'gamma': 7,
        #     'subsample': 0.7,
        #     'colsample_bytree': 0.9,
        #     'colsample_bylevel': 0.7,
        #     'reg_alpha': 10,
        #     'reg_lambda': 10,
        #     'scale_pos_weight': 1
        # }
        # log.info('默认参数 {}'.format(params))
        #
        # log.info('构建基础模型')

        if is_feature_select:
            log.info('需要进行变量筛选')
            fea_names = autobmt.feature_select({'dev': dev_data, 'nodev': nodev_data}, fea_names, self.target,
                                               feature_select_method, method_threhold,
                                               corr_threhold,
                                               psi_threhold)

        if is_auto_tune_params:
            log.info('需要进行自动调参')
            best_model = autobmt.classifiers_model_auto_tune_params(
                train_data=(dev_data[fea_names], dev_data[self.target]),
                test_data=(nodev_data[fea_names], nodev_data[self.target]))
            params = best_model.get_params()

        if is_stepwise_del_feature:
            log.info('需要逐步的删除变量')
            _, fea_names = autobmt.stepwise_del_feature({'dev': dev_data, 'nodev': nodev_data}, fea_names, self.target,
                                                        params)

        # 最终模型
        log.info('使用自动调参选出来的最优参数+筛选出来的变量，构建最终模型')
        log.info('最终变量的个数{}, 最终变量{}'.format(len(fea_names), fea_names))
        log.info('自动调参选出来的最优参数{}'.format(params))
        xgb_clf = XGBClassifier(**params)
        xgb_clf.fit(dev_data[fea_names], dev_data[self.target])

        # ###
        # pred_nodev = xgb_clf.predict_proba(nodev_data[fea_names])[:, 1]
        # pred_dev = xgb_clf.predict_proba(dev_data[fea_names])[:, 1]
        # df_pred_nodev = pd.DataFrame({'target': nodev_data[self.target], 'p': pred_nodev}, index=nodev_data.index)
        # df_pred_dev = pd.DataFrame({'target': dev_data[self.target], 'p': pred_dev}, index=dev_data.index)
        # ###

        # ###
        # df_pred_nodev = nodev_data[self.no_feature_names + fea_names]
        # df_pred_dev = dev_data[self.no_feature_names + fea_names]
        # df_pred_nodev['p'] = xgb_clf.predict_proba(df_pred_nodev[fea_names])[:, 1]
        # df_pred_dev['p'] = xgb_clf.predict_proba(df_pred_dev[fea_names])[:, 1]
        # ###

        ###
        df_pred_nodev = nodev_data[self.no_feature_names]
        df_pred_dev = dev_data[self.no_feature_names]
        df_pred_nodev['p'] = xgb_clf.predict_proba(nodev_data[fea_names])[:, 1]
        df_pred_dev['p'] = xgb_clf.predict_proba(dev_data[fea_names])[:, 1]
        ###

        # 计算auc、ks、psi
        test_ks = autobmt.get_ks(df_pred_nodev[self.target], df_pred_nodev['p'])
        train_ks = autobmt.get_ks(df_pred_dev[self.target], df_pred_dev['p'])
        test_auc = autobmt.get_auc(df_pred_nodev[self.target], df_pred_nodev['p'])
        train_auc = autobmt.get_auc(df_pred_dev[self.target], df_pred_dev['p'])

        q_cut_list = np.arange(0, 1, 1 / 20)
        bins = np.append(np.unique(np.quantile(df_pred_nodev['p'], q_cut_list)), df_pred_nodev['p'].max() + 0.1)
        df_pred_nodev['range'] = pd.cut(df_pred_nodev['p'], bins=bins, precision=0, right=False).astype(str)
        df_pred_dev['range'] = pd.cut(df_pred_dev['p'], bins=bins, precision=0, right=False).astype(str)
        nodev_psi = autobmt.psi(df_pred_nodev['range'], df_pred_dev['range'])
        res_dict = {'dev_auc': train_auc, 'nodev_auc': test_auc, 'dev_ks': train_ks, 'nodev_ks': test_ks,
                    'nodev_dev_psi': nodev_psi}
        log.info('auc & ks & psi: {}'.format(res_dict))
        log.info('*' * 30 + '自动构建模型完成！！！' + '*' * 30)

        ##############
        log.info('*' * 30 + '建模相关结果开始保存！！！' + '*' * 30)
        joblib.dump(xgb_clf.get_booster(), os.path.join(self.ml_res_save_path, 'xgb.ml'))
        joblib.dump(xgb_clf, os.path.join(self.ml_res_save_path, 'xgb_sk.ml'))
        autobmt.dump_to_pkl(xgb_clf.get_booster(), os.path.join(self.ml_res_save_path, 'xgb.pkl'))
        autobmt.dump_to_pkl(xgb_clf, os.path.join(self.ml_res_save_path, 'xgb_sk.pkl'))
        json.dump(xgb_clf.get_params(), open(os.path.join(self.ml_res_save_path, 'xgb.params'), 'w'))
        xgb_clf.get_booster().dump_model(os.path.join(self.ml_res_save_path, 'xgb.txt'))
        pd.DataFrame([res_dict]).to_csv(os.path.join(self.ml_res_save_path, 'xgb_auc_ks_psi.csv'), index=False)

        pd.DataFrame(list(xgb_clf.get_booster().get_fscore().items()),
                     columns=['fea_names', 'weight']
                     ).sort_values('weight', ascending=False).set_index('fea_names').to_csv(
            os.path.join(self.ml_res_save_path, 'xgb_featureimportance.csv'))

        nodev_data[self.no_feature_names + fea_names].head(500).to_csv(
            os.path.join(self.ml_res_save_path, 'xgb_test_input.csv'),
            index=False)

        ##############pred to score
        df_pred_nodev['score'] = df_pred_nodev['p'].map(
            lambda x: autobmt.to_score(x, self.AB['A'], self.AB['B'], self.positive_corr))
        df_pred_dev['score'] = df_pred_dev['p'].map(
            lambda x: autobmt.to_score(x, self.AB['A'], self.AB['B'], self.positive_corr))
        ##############pred to score

        df_pred_nodev.append(df_pred_dev).to_csv(os.path.join(self.ml_res_save_path, 'xgb_pred_to_report_data.csv'),
                                                 index=False)

        log.info('*' * 30 + '建模相关结果保存完成！！！保存路径为：{}'.format(self.ml_res_save_path) + '*' * 30)

        return xgb_clf, fea_names

    @classmethod
    def predict(cls, to_pred_df=None, model_path=None):
        if to_pred_df is None:
            raise ValueError('需要进行预测的数据集不能为None，请指定数据集！！！')
        if model_path is None:
            raise ValueError('模型路径不能为None，请指定模型文件路径！！！')

        try:
            model = joblib.load(os.path.join(model_path, 'xgb.ml'))
        except:
            model = pickle.load(open(os.path.join(model_path, 'xgb.pkl'), 'rb'))

        try:
            model_feature_names = model.feature_names
        except:
            model_feature_names = model.get_booster().feature_names
            model = model.get_booster()

        to_pred_df['p'] = model.predict(xgb.DMatrix(to_pred_df[model_feature_names]))

        return to_pred_df
