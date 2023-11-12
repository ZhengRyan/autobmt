#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: stepwise.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-03-06
'''
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression

from .logger_utils import Logger
from .metrics import get_auc, get_ks, AIC, BIC, MSE
from .utils import split_target, unpack_tuple, step_evaluate_models, \
    get_max_corr_feature, model_predict_evaluate

INTERCEPT_COLS = 'intercept'

warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns', None)
log = Logger(level="info", name=__name__).logger


class StatsModel:
    def __init__(self, estimator='ols', criterion='aic', intercept=False):
        if isinstance(estimator, str):
            Est = self.get_estimator(estimator)
            estimator = Est(fit_intercept=intercept, )

        self.estimator = estimator
        self.intercept = intercept
        self.criterion = criterion

    def get_estimator(self, name):
        from sklearn.linear_model import (
            LinearRegression,
            LogisticRegression,
            Lasso,
            Ridge,
        )

        ests = {
            'ols': LinearRegression,
            'lr': LogisticRegression,
            'lasso': Lasso,
            'ridge': Ridge,
        }

        if name in ests:
            return ests[name]

        raise Exception('estimator {name} is not supported'.format(name=name))

    def stats(self, X, y):
        """
        """
        X = X.copy()

        if isinstance(X, pd.Series):
            X = X.to_frame()

        self.estimator.fit(X, y)

        if hasattr(self.estimator, 'predict_proba'):
            pre = self.estimator.predict_proba(X)[:, 1]
        else:
            pre = self.estimator.predict(X)

        coef = self.estimator.coef_.reshape(-1)

        if self.intercept:
            coef = np.append(coef, self.estimator.intercept_)
            X[INTERCEPT_COLS] = np.ones(X.shape[0])

        n, k = X.shape

        t_value = self.t_value(pre, y, X, coef)
        p_value = self.p_value(t_value, n)
        c = self.get_criterion(pre, y, k)

        return {
            't_value': pd.Series(t_value, index=X.columns),
            'p_value': pd.Series(p_value, index=X.columns),
            'criterion': c
        }

    def get_criterion(self, pre, y, k):
        if self.criterion == 'aic':
            llf = self.loglikelihood(pre, y, k)
            return AIC(pre, y, k, llf=llf)

        if self.criterion == 'bic':
            llf = self.loglikelihood(pre, y, k)
            return BIC(pre, y, k, llf=llf)

        if self.criterion == 'ks':
            return get_ks(y, pre)

        if self.criterion == 'auc':
            return get_auc(y, pre)

    def t_value(self, pre, y, X, coef):
        n, k = X.shape
        mse = sum((y - pre) ** 2) / float(n - k)
        nx = np.dot(X.T, X)

        if np.linalg.det(nx) == 0:
            return np.nan

        std_e = np.sqrt(mse * (np.linalg.inv(nx).diagonal()))
        return coef / std_e

    def p_value(self, t, n):
        return stats.t.sf(np.abs(t), n - 1) * 2

    def loglikelihood(self, pre, y, k):
        n = len(y)
        mse = MSE(pre, y)
        return (-n / 2) * np.log(2 * np.pi * mse * np.e)


def stepwise(frame, target='target', estimator='ols', direction='both', criterion='aic',
             p_enter=0.01, p_remove=0.01, p_value_enter=0.2, intercept=False,
             max_iter=None, return_drop=False, exclude=None):
    """
    逐步回归选择特征
    Args:
        frame (DataFrame): 用于训练模型的数据集
        target (str): 目标变量名称
        estimator (str): 用于统计的模型
        direction (str): 前向逐步还是后向逐步, 支持“forward”、“backward”和“both”，建议“both”
        criterion (str): 统计模型的信息准则, 支持“aic”、“bic”
        p_enter (float): 阈值将在“forward”和“both”中使用，用于保留特征
        p_remove (float): 阈值将在“backward”中使用，用于剔除特征
        intercept (bool): 是否需要截距项
        p_value_enter (float): 阈值将在“both”中使用，用于剔除特征
        max_iter (int): 最大迭代次数
        return_drop (bool): 是否需要返回删除的特征
        exclude (array-like): 不参与特征筛选的特征列表

    Returns:
        DataFrame: 筛选后的数据集
        array: 删除的特征列表
    """
    df, y = split_target(frame, target)

    if exclude is not None:
        df = df.drop(columns=list(set(exclude) & set(df.columns)))

    drop_list = []
    remaining = df.columns.tolist()

    selected = []

    sm = StatsModel(estimator=estimator, criterion=criterion, intercept=intercept)

    order = -1 if criterion in ['aic', 'bic'] else 1

    best_score = -np.inf * order

    iter = -1
    while remaining:
        iter += 1
        if max_iter and iter > max_iter:
            break

        l = len(remaining)
        test_score = np.zeros(l)
        test_res = np.empty(l, dtype=object)

        if direction == 'backward':
            for i in range(l):
                test_res[i] = sm.stats(
                    df[remaining[:i] + remaining[i + 1:]],
                    y,
                )
                test_score[i] = test_res[i]['criterion']

            curr_ix = np.argmax(test_score * order)
            curr_score = test_score[curr_ix]

            if (curr_score - best_score) * order < p_remove:
                break

            name = remaining.pop(curr_ix)
            drop_list.append(name)

            best_score = curr_score

        # forward and both
        else:
            for i in range(l):
                test_res[i] = sm.stats(
                    df[selected + [remaining[i]]],
                    y,
                )
                test_score[i] = test_res[i]['criterion']

            curr_ix = np.argmax(test_score * order)
            curr_score = test_score[curr_ix]

            name = remaining.pop(curr_ix)
            if (curr_score - best_score) * order < p_enter:
                drop_list.append(name)

                # early stop
                if selected:
                    drop_list += remaining
                    break

                continue

            selected.append(name)
            best_score = curr_score

            if direction == 'both':
                p_values = test_res[curr_ix]['p_value']
                drop_names = p_values[p_values > p_value_enter].index

                for name in drop_names:
                    selected.remove(name)
                    drop_list.append(name)

    r = frame.drop(columns=drop_list)

    res = (r,)
    if return_drop:
        res += (drop_list,)

    return unpack_tuple(res)


class StepWise:
    def __init__(self, df, target='target', features=[], exclude_columns=None, iv_psi_df=None, var_bin_woe={},
                 match_dict={},
                 object='test_ks', features_corr_df=None,
                 include_columns=[], p_threshold=0.05, vif_threshold=5, max_varcnt=10, A=404.65547022, B=72.1347520444,
                 is_return_var=False):
        self.df = df
        self.target = target
        self.features = features
        self.exclude_columns = exclude_columns
        self.iv_psi_df = iv_psi_df
        self.var_bin_woe = var_bin_woe
        self.match_dict = match_dict  # 特征的中文字典名称
        self.object = object
        self.features_corr_df = features_corr_df  # 变量相关性替换列表
        self.include_columns = include_columns
        self.p_threshold = p_threshold
        self.vif_threshold = vif_threshold
        self.A = A
        self.B = B
        self.max_varcnt = len(self.features) if max_varcnt > len(self.features) else max_varcnt
        self.step_evaluate_log_df = []
        self.predict_data = None
        self.is_return_var = is_return_var

    @property
    def get_evaluate_df_log(self):
        """合并每一步的评估结果"""
        if len(self.step_evaluate_log_df) == 0:
            log.info("并未进行评估过!!!")
            return None
        else:
            evaluate_log_df = pd.concat(self.step_evaluate_log_df, axis=0).reset_index(drop=True)
            return evaluate_log_df

    @property
    def get_output_data(self):
        """获取预测的结果"""
        if self.predict_data is not None and isinstance(self.predict_data, pd.DataFrame):
            if self.is_return_var:
                for k, v in self.model.params.to_dict().items():
                    if k == 'const':
                        self.predict_data['const'] = round(self.A - self.B * v)
                    else:
                        self.predict_data["{}_fscore".format(k)] = self.predict_data[k].map(
                            lambda x: round(-(self.B * v * x)))

                # dump_to_pkl(score_card_structure,'./score_card_structure.pkl')
                in_model_features = [i for i in self.predict_data.columns.tolist() if '_fscore' in i] + ['const']
                self.predict_data['woe2score'] = self.predict_data[in_model_features].sum(axis=1).map(
                    lambda x: round(x))
                return self.predict_data
            else:
                return self.predict_data
        else:
            ValueError("并没有进行预测，请先执行stepwise_apply()!")

    def run(self):
        """
        新版双向stepwise方法
        Returns:

        """
        # 执行了最优分箱后auc,ks
        step_name = "best_binning"
        step_evaluate_df, evaluate_log = step_evaluate_models(self.df, self.features, self.target, stepname=step_name)
        self.step_evaluate_log_df.append(step_evaluate_df)
        log.info("{},evaluate --- {}".format(step_name, evaluate_log))

        log.info("开始进行Stepwise,变量个数为:{}".format(len(self.features)))
        # 前置检查
        self.check()
        log.info("Stepwise前置检查完成")
        record_features_log = []  # 记录整个stepwise的变量

        initial_list = self.include_columns.copy()  # 保留的特征
        train_data = self.df[self.df['type'] == 'train']
        test_data = self.df[self.df['type'] == 'test']
        if self.iv_psi_df is not None and isinstance(self.iv_psi_df, pd.DataFrame):
            self.features = self.iv_psi_df.sort_values(by='IV', ascending=False).index.to_list()
        high_iv_name = self.features[0]
        initial_list.append(high_iv_name)
        self.features.remove(high_iv_name)
        flag_features = initial_list.copy()  # 记录特征
        # 最大iv的目标值，作为基准
        base_object_value, _ = get_object_value_by_train_lr(train_data[initial_list], train_data[self.target],
                                                            test_data[initial_list], test_data[self.target])
        record_features_log.append(("round-0", ",".join(initial_list), base_object_value, 0))
        log.info("最高iv变量{}的基础目标值为{}".format(high_iv_name, base_object_value))
        step = 1
        while True:
            result_dict = {}
            # 前向选择过程
            # 遍历判断添加一个变量后，目标值是否有提升
            for name in self.features:
                if name not in flag_features:
                    object_value, _ = get_object_value_by_train_lr(train_data[initial_list + [name]],
                                                                   train_data[self.target],
                                                                   test_data[initial_list + [name]],
                                                                   test_data[self.target])
                    result_dict[name] = object_value
            max_key, max_value = get_max_value_in_dic(result_dict)  # 返回最优的特征和object值
            # 若有提升，则加入到候选变量中
            round_history_arr = [("rount-{}".format(step), ",".join(initial_list + [k]), v, v - base_object_value) for
                                 k, v in
                                 result_dict.items()]
            record_features_log.extend(round_history_arr)

            if max_value > base_object_value:
                initial_list.append(max_key)  # 加入到候选变量
                flag_features.append(max_key)  # 记录该特征已经选择过
                base_object_value = max_value  # 更新当前最优object值
                # 后向判断过程
                df_pvalue_vif, initial_list = calculate_features_p_value_vif(train_data, initial_list, self.target)
                filter_df = df_pvalue_vif[(df_pvalue_vif['pvalue'] > self.p_threshold) | (
                    df_pvalue_vif['vif'] > self.vif_threshold)]
                if filter_df.shape[0] > 0:
                    drop_col = filter_df.index.tolist()
                    # 删除后向剔除的特征
                    # TODO
                    initial_list = [i for i in initial_list if i not in drop_col]
                    drop_log = ("rount-{}-drop".format(step), ",".join(drop_col), 0, 0)  # 被剔除变量
                    record_features_log.append(drop_log)
                tmp = ("rount-{}-best".format(step), ",".join(initial_list), base_object_value, 0)  # 这一轮的最优变量
                record_features_log.append(tmp)

            step += 1
            # 若满足下列条件，则跳出循环
            if len(initial_list) >= self.max_varcnt or max_value - base_object_value < 0:
                break
        log.info("Stepwise目标选择完成,剩余变量个数为:{}".format(len(initial_list)))
        # 构造stepwise的过程输出和模型结果
        result_df, evaluate_df = self.output(train_data, initial_list, self.target)
        log_df = pd.DataFrame(record_features_log, columns=['round-n', 'features', 'object_value', 'diff'])
        step_name = "toad_stepwise"
        step_evaluate_df, evaluate_log = step_evaluate_models(self.df, initial_list, self.target, stepname=step_name)
        self.step_evaluate_log_df.append(step_evaluate_df)
        log.info("{},evaluate --- {}".format(step_name, evaluate_log))
        return initial_list, result_df, log_df, evaluate_df, self.get_evaluate_df_log

    def stepwise_apply(self):
        """
        基于pvalue、aic、bid的Stepwise方法
        Returns:

        """
        # 执行了最优分箱后auc,ks
        step_name = "best_binning"
        step_evaluate_df, evaluate_log = step_evaluate_models(self.df, self.features, self.target, stepname=step_name)
        self.step_evaluate_log_df.append(step_evaluate_df)
        log.info("{},evaluate --- {}".format(step_name, evaluate_log))

        train_data = self.df[self.df['type'] == 'train']
        if self.features_corr_df is not None and isinstance(self.features_corr_df, pd.DataFrame):
            max_corr_df = self.features_corr_df  # 如果相关性df传入了，则用传入的
        else:  # 否则用woe转换后的
            max_corr_df = get_max_corr_feature(train_data, self.features)  # 计算每个变量相关性最高的变量
        tmp_df = train_data[self.features + [self.target]]
        final_data = stepwise(tmp_df, target=self.target, estimator='ols', direction='both', criterion='aic',
                              p_enter=0.05, p_remove=0.05, )

        selected_features = [name for name in final_data.columns.to_list() if name not in self.exclude_columns]
        log.info("Stepwise目标选择完成,剩余变量个数为:{}".format(len(selected_features)))
        log.info("Stepwise保留变量为:{}".format(",".join(selected_features)))
        result_df, evaluate_df = self.output(train_data, selected_features, self.target)  # 训练模型
        step_name = "toad_stepwise"
        step_evaluate_df, evaluate_log = step_evaluate_models(self.df, selected_features, self.target,
                                                              stepname=step_name)
        self.step_evaluate_log_df.append(step_evaluate_df)
        log.info("{},evaluate --- {}".format(step_name, evaluate_log))

        # 关联上相关性最强的特征,方便替换
        result_df = result_df.merge(max_corr_df, how='left', left_index=True, right_index=True)
        if len(self.match_dict) > 0:
            result_df.insert(8, 'corr_name_cn', result_df['corr_name'].map(self.match_dict))
        result_df = result_df.reset_index()

        ##TODO 考虑在result_df进行psi、vif、相关性的过滤。result_df是10.model_stepwise

        return selected_features, result_df, evaluate_df, self.get_evaluate_df_log

    def output(self, train_data, features, target):
        '''

        Args:
            train_data:
            features:
            target:

        Returns:

        '''
        if features is None:
            raise ValueError("没有入模变量")

        # 训练模型,计算pvalue,vif
        df_pvalue_vif_coef, model, features = calculate_features_p_value_vif(train_data, features, target,
                                                                             need_coef=True,
                                                                             need_model=True)
        self.model = model
        if len(self.match_dict) > 0:
            df_pvalue_vif_coef.insert(0, 'cn', df_pvalue_vif_coef.index.map(self.match_dict))
        # 评估模型
        evaluate_df, self.predict_data = model_predict_evaluate(model, self.df, features, self.target, self.A, self.B,
                                                                self.exclude_columns, self.is_return_var)
        if self.iv_psi_df is not None:
            df_iv_psi = self.iv_psi_df[self.iv_psi_df.index.isin(features)]
            result_df = df_pvalue_vif_coef.merge(df_iv_psi, how='left', left_index=True, right_index=True)
            return result_df, evaluate_df
        else:
            return df_pvalue_vif_coef, evaluate_df

    def get_scorecard_structure(self):
        score_card_structure = {}
        for k, v in self.model.params.to_dict().items():
            # scorecard
            if k == 'const':
                score_card_structure['const'] = ['-', round(self.A - self.B * v)]
            else:
                one_feature_binning = self.var_bin_woe[k]
                one_feature_dict = {}
                for k1, v1 in one_feature_binning.items():
                    one_feature_dict[k1] = [float(v1), round(-(self.B * v * float(v1)))]  # v1是woe值
                score_card_structure[k] = one_feature_dict
        return score_card_structure

    def check(self):
        """
        特征选择模块，前置检查,符合要求，则往下运行
        Returns:

        """
        if len(self.features) <= 0 or self.features is None:
            raise ValueError("入模型变量不能为空，请检查!!!")
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError("数据集不能为空并且数据集必须是dataframe!!!")
        if self.target is None:
            raise ValueError("数据集的目标变量名称不能为空!!!")
        if self.exclude_columns is None or "type" not in self.exclude_columns or self.target not in self.exclude_columns:
            raise ValueError("exclude 不能为空，必须包含target,type字段!!!")

    @staticmethod
    def scorecard_to_excel(scorecard, workbook, sheet_name):
        # excel相关格式
        # workbook = xlsxwriter.Workbook(output, {'nan_inf_to_errors': True})
        # worksheet = workbook.add_worksheet('分箱详情')
        worksheet = workbook.add_worksheet(sheet_name)
        # Add a header format.
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'center',
            'font_name': 'Consolas',
            'font_size': 11,
            'border': 1,
            'bottom': 2,
        })
        font_formats = workbook.add_format({'font_name': 'Consolas', 'font_size': 11, })  # 其他

        def write_to_excel_by_dict(scorecard, worksheet, start_row, start_column):
            for featurename, interval_value in scorecard.items():
                if isinstance(interval_value, dict):
                    worksheet.write(start_row, start_column, featurename, font_formats)
                    for interval, value in interval_value.items():
                        woe, score = value[0], value[1]
                        worksheet.write(start_row, start_column + 1, interval, font_formats)
                        worksheet.write(start_row, start_column + 2, woe, font_formats)
                        worksheet.write(start_row, start_column + 3, score, font_formats)
                        start_row += 1
                elif isinstance(interval_value, list):
                    nothing, score = interval_value[0], interval_value[1]
                    worksheet.write(start_row, start_column, featurename, font_formats)
                    worksheet.write(start_row, start_column + 1, nothing, font_formats)
                    worksheet.write(start_row, start_column + 2, nothing, font_formats)
                    worksheet.write(start_row, start_column + 3, score, font_formats)
                    start_row += 1

        # 写入标题
        for col_num, value in enumerate(['特征名称', '特征区间', '特征区间WOE值', '特征区间得分']):
            worksheet.write(0, col_num, value, header_format)

        # 写入数据
        start_row, start_column = 1, 0
        write_to_excel_by_dict(scorecard, worksheet, start_row, start_column)
        worksheet.set_column(0, 1, 20)


def get_object_value_by_train_lr(X_train, y_train, X_test, y_test, object_func='default'):
    """
    计算目标值
    Args:
        X_train:
        y_train:
        X_test:
        y_test:
        object_func:

    Returns:

    """

    lr = LogisticRegression().fit(X_train, y_train)
    y_test_pred = lr.predict_proba(X_test)[:, 1]
    test_ks = get_ks(y_test, y_test_pred)
    test_auc = get_auc(y_test, y_test_pred)

    if object_func == "test_ks":
        return test_ks
    elif object_func == "test_auc":
        return test_auc
    elif object_func == "default":
        return test_ks, test_auc


def get_max_value_in_dic(dic):
    """
    返回字典中最大value的k,v
    Args:
        dic:

    Returns:

    """
    import operator
    key = max(dic.items(), key=operator.itemgetter(1))[0]
    return key, dic[key]


def calculate_features_p_value_vif(df, features, target, just_pvalue=False, need_coef=False, need_model=False):
    """
    计算每个变量的p-value和vif和coef
    Args:
        df:
        features:
        target:
        just_pvalue:
        need_coef:
        need_model:

    Returns:

    """
    import autobmt

    index = 1
    while True:  # 循环的目的是保证入模变量的系数都为整
        model = sm.Logit(df[target], sm.add_constant(df[features])).fit()
        ignore_features = [k for k, v in model.params.to_dict().items() if k != 'const' and v < 0]
        if len(ignore_features) == 0:
            break
        features = [i for i in features if i not in ignore_features]
        index += 1
    df_pvalue = pd.DataFrame(model.pvalues, columns=['pvalue'])
    if just_pvalue:
        return df_pvalue
    df_coef = pd.DataFrame(model.params, columns=['coef'])
    df_vif = pd.Series(index=features)
    for col in features:
        df_vif[col] = pd.Series(autobmt.get_vif(df[list(set(features) - set([col]))], df[col]))
    df_vif = pd.DataFrame(df_vif, columns=['vif'])
    if need_coef:
        df_pvalue_vif_coef = df_coef.merge(df_pvalue, how='left', left_index=True, right_index=True) \
            .merge(df_vif, how='left', left_index=True, right_index=True)
        if need_model:
            return df_pvalue_vif_coef, model, features
        else:
            return df_pvalue_vif_coef, features
    else:
        df_pvalue_vif = df_pvalue.merge(df_vif, how='inner', left_index=True, right_index=True)
        return df_pvalue_vif, features
