#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: bayes_opt_tuner.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-25
'''

import time
import warnings

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .metrics import get_ks, get_auc
from .utils import get_accuracy, get_recall, get_precision, get_f1, r2

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class ModelTune():
    def __init__(self):
        self.base_model = None
        self.best_model = None
        self.tune_params = None
        self.loss = np.inf
        self.default_params = None
        self.int_params = None
        self.init_params = None
        self.metrics = None
        self._metrics_score = []
        self.scores = []

    def get_model(self):
        return self.best_model

    def _map_metrics(self):
        mapper = {
            'accuracy': get_accuracy,
            'f1': get_f1,
            'precision': get_precision,
            'recall': get_recall,
            'r2': r2,
            'auc': get_auc,
            'ks': get_ks
        }

        for metric in self.metrics:
            if metric not in mapper:
                raise ValueError('指定的指标 ''`{}` 不支持'.format(metric))
            self._metrics_score.append(mapper[metric])
            self.scores.append(0.)

    def fit(self, train_data=(), test_data=()
            , init_points=30, iterations=120, metrics=[]):
        '''

        Args:
            train_data:
            test_data:
            init_points:
            iterations:
            metrics:

        Returns:

        '''

        if len(metrics) > 0:
            self.metrics = metrics
        self._map_metrics()

        X_train, y_train = train_data
        X_test, y_test = test_data

        def loss_fun(train_result, test_result, weight=0.3):

            # return test_result - 2 ** abs(test_result - train_result)
            return test_result - 2 ** abs(test_result - train_result) * weight

        # def loss_fun(train_result, test_result):
        #     train_result = train_result * 100
        #     test_result = test_result * 100
        #
        #     return train_result - 2 ** abs(train_result - test_result)

        def obj_fun(**params):
            for param in self.int_params:
                params[param] = int(round(params[param]))

            model = self.base_model(**params, **self.default_params)
            model.fit(X_train, y_train)

            pred_test = model.predict_proba(X_test)[:, 1]
            pred_train = model.predict_proba(X_train)[:, 1]

            # test_auc = get_auc(y_test, pred_test)
            # train_auc = get_auc(y_train, pred_train)
            # print('test_auc is : ', test_auc)
            # print('train_auc is : ', train_auc)

            test_ks = get_ks(y_test, pred_test)
            train_ks = get_ks(y_train, pred_train)
            # print('test_ks is : ', test_ks)
            # print('train_ks is : ', train_ks)

            # maximize = loss_fun(train_auc, test_auc)
            maximize = loss_fun(train_ks, test_ks)
            # print('max_result is : ', maximize)
            # max_result = loss_fun(train_ks, test_ks) * 2 + loss_fun(train_auc, test_auc)

            loss = -maximize
            if loss < self.loss:
                self.loss = loss
                self.best_model = model
                # print('best model result is {}'.format(loss))
                # print('best model params is : ')
                # print(self.best_model.get_params())
                for i, _metric in enumerate(self._metrics_score):
                    self.scores[i] = _metric(y_test, pred_test)
            # print('current obj_fun result is : ', maximize)

            return maximize

        params_optimizer = BayesianOptimization(obj_fun, self.tune_params, random_state=1)
        log.info('需要优化的超参数是 : {}'.format(params_optimizer.space.keys))

        log.info('开始优化超参数!!!')
        start = time.time()

        params_optimizer.maximize(init_points=1, n_iter=0, acq='ei',
                                  xi=0.0)
        params_optimizer.probe(self.init_params, lazy=True)

        # params_optimizer.probe(self.init_params, lazy=True)

        params_optimizer.maximize(init_points=0, n_iter=0)

        params_optimizer.maximize(init_points=init_points, n_iter=iterations, acq='ei',
                                  xi=0.0)  # init_points：探索开始探索之前的迭代次数；iterations：方法试图找到最大值的迭代次数
        # params_optimizer.maximize(init_points=init_points, n_iter=iterations, acq='ucb', xi=0.0, alpha=1e-6)
        end = time.time()
        log.info('优化参数结束!!! 共耗时{} 分钟'.format((end - start) / 60))
        log.info('最优参数是 : {}'.format(params_optimizer.max['params']))
        log.info('{} model 最大化的结果 : {}'.format(type(self.best_model), params_optimizer.max['target']))


class ClassifierModel(ModelTune):
    def __init__(self):
        super().__init__()
        self.metrics = ['auc', 'ks']


class RegressorModel(ModelTune):
    def __init__(self):
        super().__init__()
        self.metrics = ['r2', 'rmse']


class XGBClassifierTuner(ClassifierModel):
    def __init__(self):
        super().__init__()

        self.base_model = XGBClassifier
        self.tune_params = {
            'learning_rate': (0.01, 0.15),
            'n_estimators': (90, 300),
            'max_depth': (2, 7),
            'min_child_weight': (1, 300),
            'subsample': (0.4, 1.0),
            'colsample_bytree': (0.3, 1.0),
            'colsample_bylevel': (0.5, 1.0),
            'gamma': (0, 20.0),
            'reg_alpha': (0, 20.0),
            'reg_lambda': (0, 20.0),
            # 'scale_pos_weight': (1, 5),
            # 'max_delta_step': (0, 10)
        }

        self.default_params = {
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'nthread': -1
        }

        self.init_params = {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 3,
            'min_child_weight': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.9,
            'colsample_bylevel': 0.7,
            'gamma': 7,
            'reg_alpha': 10,
            'reg_lambda': 10,
            # 'scale_pos_weight': 1
        }

        self.int_params = ['max_depth', 'n_estimators']


class LGBClassifierTuner(ClassifierModel):
    '''
    英文版：https://lightgbm.readthedocs.io/en/latest/Parameters.html

    中文版：https://lightgbm.apachecn.org/#/docs/6

    其他注解：https://medium.com/@gabrieltsen
    '''

    def __init__(self):
        super().__init__()

        self.base_model = LGBMClassifier
        self.tune_params = {
            'max_depth': (3, 15),
            'num_leaves': (16, 128),
            'learning_rate': (0.01, 0.2),
            'reg_alpha': (0, 100),
            'reg_lambda': (0, 100),
            'min_child_samples': (1, 100),
            'min_child_weight': (0.01, 100),
            'colsample_bytree': (0.5, 1),
            'subsample': (0.5, 1),
            'subsample_freq': (2, 10),
            'n_estimators': (90, 500),

        }

        self.default_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 1024,
            'n_jobs': -1,
            'num_threads': -1,
            'verbose': -1,

        }

        self.init_params = {
            'max_depth': -1,
            "num_leaves": 31,
            "learning_rate": 0.02,
            "reg_alpha": 0.85,
            "reg_lambda": 3,
            "min_child_samples": 20,  # TODO注意修改 .......sklearn:min_child_samples    原生:min_data、min_data_in_leaf
            "min_child_weight": 0.05,  # sklearn:min_child_weight    原生:min_hessian、min_sum_hessian_in_leaf
            "colsample_bytree": 0.9,  # sklearn:colsample_bytree   原生:feature_fraction
            "subsample": 0.8,  # sklearn:subsample  原生:bagging_fraction
            "subsample_freq": 2,  # sklearn:subsample_freq  原生:bagging_freq
            "n_estimators": 100  # sklearn:n_estimators  原生:num_boost_round、num_iterations
        }

        self.int_params = ['max_depth', 'num_leaves', 'min_child_samples', 'n_estimators', 'subsample_freq']


classifiers_dic = {
    # 'lr': LogisticRegressionTuner,
    # 'rf': RandomForestClassifierTuner,
    'xgb': XGBClassifierTuner,
    'lgb': LGBClassifierTuner
}


def classifiers_model_auto_tune_params(models=['xgb'], metrics=[], train_data=(), test_data=()
                                       , init_points=30, iterations=120, verbose=1):
    '''

    Args:
        models:
        metrics:
        train_data:
        test_data:
        init_points:
        iterations:
        verbose:

    Returns:

    '''
    best_model = None
    if not isinstance(models, list):
        raise AttributeError('models参数必须是一个列表, ', '但实际是 {}'.format(type(models)))
    if len(models) == 0:
        models = list(classifiers_dic.keys())
    classifiers = []
    for model in models:
        if model in classifiers_dic:
            classifiers.append(classifiers_dic[model])
    loss = np.inf
    _model = None
    for classifier in classifiers:
        if verbose:
            log.info("优化 {}...".format(classifier()))
        _model = classifier()
        _model.fit(train_data=train_data,
                   test_data=test_data
                   , init_points=init_points, iterations=iterations, metrics=metrics)

        _loss = _model.loss
        if verbose:
            _show_fit_log(_model)
        if _loss < loss:
            loss = _loss
            best_model = _model

    return best_model.get_model()


def _show_fit_log(model):
    _out = '  最优结果: '
    _out += ' loss: {:.3}'.format(model.loss)
    _out += ' 测试集 '
    for i, _metric in enumerate(model.metrics):
        _out += ' {}: {:.3}'.format(_metric[:3],
                                    model.scores[i])
    log.info(_out)


if __name__ == '__main__':
    X = pd.read_pickle('X_train.pkl')
    X = pd.DataFrame(X)
    y = pd.read_pickle('y_train.pkl')
    y = pd.Series(y)
    X_test = pd.read_pickle('X_test.pkl')
    X_test = pd.DataFrame(X_test)
    y_test = pd.read_pickle('y_test.pkl')
    y_test = pd.Series(y_test)

    ####build model
    best_model = classifiers_model_auto_tune_params(train_data=(X, y), test_data=(X_test, y_test), verbose=1,
                                                    init_points=1,
                                                    iterations=2)
    # best_model = classifiers_model_auto_tune_params(train_data=(X, y), test_data=(X_test, y_test), verbose=1)
    print('classifiers_model run over!!!')
    print(type(best_model))
    print(best_model.get_params())
    train_pred_y = best_model.predict_proba(X)[:, 1]
    test_pred_y = best_model.predict_proba(X_test)[:, 1]
    ####build model

    #####build model
    # best_model = LGBMClassifier()
    # best_model.fit(X,y)
    # print(best_model.get_params())
    # train_pred_y = best_model.predict_proba(X)[:, 1]
    # test_pred_y = best_model.predict_proba(X_test)[:, 1]
    #####build model

    # #####build model
    # import lightgbm as lgb
    #
    # init_params = {
    #     "boosting_type": "gbdt",
    #     "objective": "binary",
    #     "metric": "auc",
    # }
    # best_model = lgb.train(params=init_params, train_set=lgb.Dataset(X, y), valid_sets=lgb.Dataset(X_test, y_test))
    # best_model.save_model('lgb.txt')
    # json_model = best_model.dump_model()
    # import json
    #
    # with open('lgb.json', 'w') as f:
    #     json.dump(json_model, f)
    # train_pred_y = best_model.predict(X)
    # test_pred_y = best_model.predict(X_test)
    # #####build model

    train_auc = get_auc(y, train_pred_y)
    test_auc = get_auc(y_test, test_pred_y)
    train_ks = get_ks(y, train_pred_y)
    test_ks = get_ks(y_test, test_pred_y)
    print('train_auc is : ', train_auc, 'test_auc is : ', test_auc)
    print('train_ks is : ', train_ks, 'test_ks is : ', test_ks)

    # # #####build model
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
    #
    # clf = XGBClassifier(**params)
    # clf.fit(X, y)
    # estimator = clf.get_booster()
    # temp = estimator.save_raw()[4:]
    # # #####build model

    ####构建数据
    # from sklearn.datasets import make_classification
    # from sklearn.model_selection import train_test_split
    # import pickle
    # X, y = make_classification(n_samples=10000, random_state=1024)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #
    # with open('X_train.pkl', 'wb') as f:
    #     f.write(pickle.dumps(X_train))
    # with open('y_train.pkl', 'wb') as f:
    #     f.write(pickle.dumps(y_train))
    # with open('X_test.pkl', 'wb') as f:
    #     f.write(pickle.dumps(X_test))
    # with open('y_test.pkl', 'wb') as f:
    #     f.write(pickle.dumps(y_test))
