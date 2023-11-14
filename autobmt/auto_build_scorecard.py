#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: auto_build_scorecard.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-20
'''

import os
import time
import warnings

import numpy as np
import pandas as pd
import xlsxwriter

import autobmt

warnings.filterwarnings('ignore')

log = autobmt.Logger(level='info', name=__name__).logger


class AutoBuildScoreCard:

    def __init__(self, datasets, fea_names, target, key='key', data_type='type',
                 no_feature_names=['key', 'target', 'apply_time', 'type'], ml_res_save_path='./model_result', data_dict=None):

        if data_type not in datasets:
            raise KeyError('train、test数据集标识的字段名不存在！或未进行数据集的划分，请将数据集划分为train、test！！！')

        datasets[data_type] = datasets[data_type].map(str.lower)

        self.data_type_ar = np.unique(datasets[data_type])
        if 'train' not in self.data_type_ar:
            raise KeyError("""没有开发样本，数据集标识字段{}没有`train`该取值！！！""".format(data_type))

        if 'test' not in self.data_type_ar:
            raise KeyError("""没有验证样本，数据集标识字段{}没有`test`该取值！！！""".format(data_type))

        if target not in datasets:
            raise KeyError('样本中没有目标变量y值！！！')

        if data_dict is not None and isinstance(data_dict, pd.DataFrame):
            if not data_dict.columns.isin(['feature', 'cn']).all():
                raise KeyError("原始数据字典中没有feature或cn字段，请保证同时有feature字段和cn字段")

        # fea_names = [i for i in fea_names if i != key and i != target]
        fea_names = [i for i in fea_names if i not in no_feature_names]
        log.info('数据集变量个数 : {}'.format(len(fea_names)))
        log.info('fea_names is : {}'.format(fea_names))

        self.datasets = datasets
        self.fea_names = fea_names
        self.target = target
        self.key = key
        self.no_feature_names = no_feature_names
        self.data_dict = data_dict
        self.ml_res_save_path = ml_res_save_path + '/' + time.strftime('%Y%m%d%H%M%S_%s', time.localtime())

        os.makedirs(self.ml_res_save_path, exist_ok=True)

    def fit(self, empty_threhold=0.9, iv_threhold=0.02, corr_threhold=0.7, psi_threhold=0.1,
            dev_nodev_iv_diff_threhold=0.08):
        log.info('样本数据集类型：{}'.format(self.data_type_ar))
        log.info('样本行列情况：{}'.format(self.datasets.shape))
        log.info('样本正负占比情况：')
        log.info(self.datasets[self.target].value_counts() / len(self.datasets))

        excel_file_name_path = "{}/build_model_log.xlsx".format(self.ml_res_save_path)
        workbook = xlsxwriter.Workbook(excel_file_name_path, {'nan_inf_to_errors': True})
        excel_utils = autobmt.Report2Excel(excel_file_name_path, workbook=workbook)
        log.info('读取样本&特征数据集：{} |为样本数据，其他为特征数据'.format(self.no_feature_names))
        # =========================读取数据集结束=========================

        time_start = time.time()

        log.info('Step 1: EDA，整体数据探索性数据分析')
        all_data_eda = autobmt.detect(self.datasets)
        all_data_eda.to_excel('{}/all_data_eda.xlsx'.format(self.ml_res_save_path))
        excel_utils.df_to_excel(all_data_eda, '1.EDA')

        log.info('Step 2: 特征粗筛选开始')
        # 进行特征初步选择
        # 分箱方法支持 dt、chi、equal_freq、kmeans
        fs_dic = {
            "empty": {'threshold': empty_threhold},
            # "const": {'threshold': 0.95},
            "psi": {'threshold': psi_threhold},
            "iv": {'threshold': iv_threhold},
            # "iv_diff": {'threshold': dev_nodev_iv_diff_threhold},
            "corr": {'threshold': corr_threhold},

        }

        # TODO 考虑将self.datasets更换为train_data
        fs = autobmt.FeatureSelection(df=self.datasets, target=self.target, exclude_columns=self.no_feature_names,
                                      match_dict=self.data_dict,
                                      params=fs_dic)
        #selected_df, selected_features, select_log_df, selection_evaluate_log_df, fbfb = fs.select()
        selected_df, selected_features, select_log_df, fbfb = fs.select()
        summary = autobmt.calc_var_summary(
            fbfb.transform(selected_df[selected_df['type'] == 'train'])[selected_features + [self.target]],
            fbfb.export(), target=self.target, need_bin=False)
        # summary = fs.get_binning_summary  # 获取分箱详情
        autobmt.del_df(self.datasets)

        log.info('Step 2: 特征粗筛选结束')

        log.info('Step 3: 对剩下的变量调用最优分箱')
        log.info('剩下的变量个数 : '.format(len(selected_features)))
        log.info('剩下的变量 : '.format(selected_features))

        # train_data = selected_df[selected_df['type'] == 'train']

        # 训练集
        fb, best_binning_result = autobmt.best_binning(selected_df[selected_df['type'] == 'train'],
                                                       x_list=selected_features,
                                                       target=self.target)
        df_bin = fb.transform(selected_df, labels=False)
        train_bin = df_bin[df_bin['type'] == 'train']
        test_bin = df_bin[df_bin['type'] == 'test']
        if 'oot' in self.data_type_ar:
            oot_bin = df_bin[df_bin['type'] == 'oot']

        train_var_summary = autobmt.calc_var_summary(train_bin[selected_features + [self.target]], fb.export(),
                                                     target=self.target, need_bin=False)

        # 测试集
        # test_bin = fb.transform(test_data, labels=False)
        test_var_summary = autobmt.calc_var_summary(test_bin[selected_features + [self.target]], fb.export(),
                                                    target=self.target, need_bin=False)

        if 'oot' in self.data_type_ar:
            # oot_bin = fb.transform(oot_data, labels=False)
            oot_var_summary = autobmt.calc_var_summary(oot_bin[selected_features + [self.target]], fb.export(),
                                                       target=self.target, need_bin=False)

        # 计算拐点
        train_data_inflection_df = autobmt.compare_inflection_point(train_var_summary)
        train_data_inflection_df.columns = train_data_inflection_df.columns.map(lambda x: 'train_' + x)

        test_data_inflection_df = autobmt.compare_inflection_point(test_var_summary)
        test_data_inflection_df.columns = test_data_inflection_df.columns.map(lambda x: 'test_' + x)
        data_inflection_df = pd.concat([train_data_inflection_df, test_data_inflection_df], axis=1)
        if 'oot' in self.data_type_ar:
            oot_data_inflection_df = autobmt.compare_inflection_point(oot_var_summary)
            oot_data_inflection_df.columns = oot_data_inflection_df.columns.map(lambda x: 'oot_' + x)
            data_inflection_df = pd.concat([data_inflection_df, oot_data_inflection_df], axis=1)

        # 存储特征粗筛内容
        # 存储最优分箱内容
        excel_utils.df_to_excel(select_log_df, '2.feature_selection')
        autobmt.var_summary_to_excel(summary, workbook, '3.binning_summary')
        autobmt.var_summary_to_excel(train_var_summary, workbook, '4.best_train_binning_summary')
        autobmt.var_summary_to_excel(test_var_summary, workbook, '5.best_test_binning_summary')
        if 'oot' in self.data_type_ar:
            autobmt.var_summary_to_excel(oot_var_summary, workbook, '6.best_oot_binning_summary')
        excel_utils.df_to_excel(best_binning_result, '7.best_binning_log')
        excel_utils.df_to_excel(data_inflection_df.reset_index(), '8.data_inflection_point')
        # 转woe
        log.info('Step 4: 对剩下的变量进行woe转换')
        woetf = autobmt.WoeTransformer()
        train_woe = woetf.fit_transform(train_bin, train_bin[self.target], exclude=self.no_feature_names)
        test_woe = woetf.transform(test_bin)
        if 'oot' in self.data_type_ar:
            oot_woe = woetf.transform(oot_bin)

        # if 'oot' in self.data_type_ar:
        #     bin_data = pd.concat([train_bin, test_bin, oot_bin])
        # else:
        #     bin_data = pd.concat([train_bin, test_bin])
        # dump_model_to_file(bin_data, '{}/bin_data.pkl'.format(self.ml_res_save_path))

        #log.info('Step 5: 获取变量的iv和psi')

        # TODO list：考虑每月和总体的psi比较
        # # 计算变量的iv和psi以及missing rate
        # psi_v, frame = psi(train_woe[selected_features], test_woe[selected_features], return_frame=True)
        # iv_psi_df = pd.DataFrame(train_var_summary.groupby('var_name')['IV'].mean()).merge(pd.DataFrame(psi_v),
        #                                                                                    left_index=True,
        #                                                                                    right_index=True).rename(
        #     columns={0: 'PSI'})
        # select_log_df.set_index('feature', inplace=True)
        # iv_psi_missing_df = select_log_df[select_log_df.index.isin(selected_features)][['miss_rate']].merge(iv_psi_df,
        #                                                                                                     how='right',
        #                                                                                                     left_index=True,
        #                                                                                                     right_index=True)

        # TODO list：iv_psi_missing_df再加上coef、pvalue、vif
        # TODO list：1、通过稳定性筛选特征 2、由于分箱转woe值后，变量之间的共线性会变强，通过相关性再次筛选特征

        log.info('Step 5: 对woe转换后的变量进行stepwise')
        if 'oot' in self.data_type_ar:
            in_model_data = pd.concat([train_woe, test_woe, oot_woe])
        else:
            in_model_data = pd.concat([train_woe, test_woe])

        # dump_model_to_file(in_model_data, '{}/in_model_data.pkl'.format(self.ml_res_save_path))
        var_bin_woe = autobmt.fea_woe_dict_format(woetf.export(), fb.export())
        # sw = StepWise(in_model_data, self.target, selected_features, self.no_feature_names, iv_psi_missing_df,
        #               var_bin_woe, {})
        # selected_features, stepwise_df, evaluate_df, stepwise_evaluate_log_df = sw.stepwise_apply()
        # # selected_features, stepwise_df,stepwise_log_df, evaluate_df, stepwise_evaluate_log_df = sw.run()

        # 将woe转化后的数据做逐步回归
        final_data = autobmt.stepwise(train_woe, target=self.target, estimator='ols', direction='both',
                                      criterion='aic',
                                      exclude=self.no_feature_names)

        # # 将选出的变量应用于test/oot数据
        # final_test = test_woe[final_data.columns]
        # if 'oot' in self.data_type_ar:
        #     final_oot = oot_woe[final_data.columns]
        # print(final_data.shape)  # 逐步回归从31个变量中选出了10个

        # 确定建模要用的变量
        selected_features = [fea for fea in final_data.columns if fea not in self.no_feature_names]

        log.info('Step 6: 用逻辑回归构建模型')
        # 用逻辑回归建模
        from sklearn.linear_model import LogisticRegression

        # lr = LogisticRegression()
        # lr.fit(final_data[selected_features], final_data[self.target])
        while True:  # 循环的目的是保证入模变量的系数都为整
            lr = LogisticRegression()
            lr.fit(final_data[selected_features], final_data[self.target])
            drop_var = np.array(selected_features)[np.where(lr.coef_ < 0)[1]]
            if len(drop_var) == 0:
                break
            selected_features = list(set(selected_features) - set(drop_var))

        # # 预测训练和隔月的OOT
        # pred_train = lr.predict_proba(final_data[selected_features])[:, 1]
        # pred_test = lr.predict_proba(final_test[selected_features])[:, 1]
        # if 'oot' in self.data_type_ar:
        #     pred_oot = lr.predict_proba(final_oot[selected_features])[:, 1]
        # # 预测训练和隔月的OOT

        in_model_data['p'] = lr.predict_proba(in_model_data[selected_features])[:, 1]
        # dump_model_to_file(in_model_data[self.no_feature_names + selected_features + ['p']],
        #                    '{}/lr_in_model_data.pkl'.format(self.ml_res_save_path))

        ###
        psi_v = autobmt.psi(test_woe[selected_features], train_woe[selected_features])
        psi_v.name = 'train_test_psi'
        train_iv = train_var_summary[['var_name', 'IV']].rename(columns={'IV': 'train_iv'}).drop_duplicates().set_index(
            'var_name')
        test_iv = train_var_summary[['var_name', 'IV']].rename(columns={'IV': 'test_iv'}).drop_duplicates().set_index(
            'var_name')
        var_miss = select_log_df[['feature', 'cn', 'miss_rate']].drop_duplicates().set_index('feature')

        if 'oot' in self.data_type_ar:
            psi_o = autobmt.psi(oot_woe[selected_features], train_woe[selected_features])
            psi_o.name = 'train_oot_psi'
            oot_iv = train_var_summary[['var_name', 'IV']].rename(columns={'IV': 'oot_iv'}).drop_duplicates().set_index(
                'var_name')

        coef_s = {}
        for idx, key in enumerate(selected_features):
            coef_s[key] = lr.coef_[0][idx]
        var_coef = pd.Series(coef_s, name='coef')
        var_vif = autobmt.get_vif(final_data[selected_features])
        statsm = autobmt.StatsModel(estimator='ols', intercept=True)
        t_p_c_value = statsm.stats(final_data[selected_features], final_data[self.target])
        p_value = pd.Series(t_p_c_value['p_value'], name='p_value')
        t_value = pd.Series(t_p_c_value['t_value'], name='t_value')

        if 'oot' in self.data_type_ar:
            var_info = pd.concat(
                [var_coef, p_value, t_value, var_vif, train_iv, test_iv, oot_iv, psi_v, psi_o, var_miss],
                axis=1).dropna(subset=['coef'])
        else:
            var_info = pd.concat([var_coef, p_value, t_value, var_vif, train_iv, test_iv, psi_v, var_miss],
                                 axis=1).dropna(subset=['coef'])
        ###

        log.info('Step 7: 构建评分卡')
        card = autobmt.ScoreCard(
            combiner=fb,
            transer=woetf,
            # class_weight = 'balanced',
            # C=0.1,
            # base_score = 600,
            # base_odds = 15 ,
            # pdo = 50,
            # rate = 2
        )

        card.fit(final_data[selected_features], final_data[self.target])

        log.info('Step 8: 持久化模型，分箱点，woe值，评分卡结构======>开始')
        autobmt.dump_to_pkl(fb, '{}/fb.pkl'.format(self.ml_res_save_path))
        autobmt.dump_to_pkl(woetf, '{}/woetf.pkl'.format(self.ml_res_save_path))
        autobmt.dump_to_pkl(selected_features, '{}/in_model_var.pkl'.format(self.ml_res_save_path))

        woetf.export(to_json='{}/var_bin_woe.json'.format(self.ml_res_save_path))
        woetf.export(to_json='{}/var_bin_woe_format.json'.format(self.ml_res_save_path), var_bin_woe=var_bin_woe)
        fb.export(to_json='{}/var_split_point.json'.format(self.ml_res_save_path), bin_format=False)
        fb.export(to_json='{}/var_split_point_format.json'.format(self.ml_res_save_path))
        card.export(to_json='{}/scorecard.json'.format(self.ml_res_save_path))

        woetf.export(to_csv='{}/var_bin_woe.csv'.format(self.ml_res_save_path))
        woetf.export(to_csv='{}/var_bin_woe_format.csv'.format(self.ml_res_save_path), var_bin_woe=var_bin_woe)
        fb.export(to_csv='{}/var_split_point.csv'.format(self.ml_res_save_path), bin_format=False)
        fb.export(to_csv='{}/var_split_point_format.csv'.format(self.ml_res_save_path))
        scorecard_structure = card.export(to_dataframe=True)
        scorecard_structure.to_csv('{}/scorecard.csv'.format(self.ml_res_save_path), index=False)

        # dump_model_to_file(sw.model, '{}/lr_sm_model.pkl'.format(self.ml_res_save_path))
        # output_model_data = sw.get_output_data
        # scorecard_structure = sw.get_scorecard_structure()

        autobmt.dump_to_pkl(lr, '{}/lrmodel.pkl'.format(self.ml_res_save_path))
        autobmt.dump_to_pkl(card, '{}/scorecard.pkl'.format(self.ml_res_save_path))
        in_model_data['score'] = in_model_data['p'].map(autobmt.to_score)
        output_report_data = in_model_data[self.no_feature_names + ['p', 'score']]
        # in_model_data['card_score'] = card.predict(self.datasets)
        # output_report_data = in_model_data[self.no_feature_names + ['p', 'score', 'card_score']]

        # dump_model_to_file(output_report_data, '{}/output_report_data.pkl'.format(self.ml_res_save_path))
        output_report_data.to_csv('{}/lr_pred_to_report_data.csv'.format(self.ml_res_save_path), index=False)  #
        selected_df[self.no_feature_names + selected_features].head(500).to_csv(
            '{}/lr_test_input.csv'.format(self.ml_res_save_path),
            index=False)
        lr_auc_ks_psi = autobmt.get_auc_ks_psi(output_report_data)
        lr_auc_ks_psi.to_csv('{}/lr_auc_ks_psi.csv'.format(self.ml_res_save_path),
                                                          index=False)
        # stepwise_df.reset_index().to_csv('{}/feature_coef_pvalue_IV_vif_psi_corr.csv'.format(self.ml_res_save_path),
        #                                  index=False)
        # dump_model_to_file(scorecard_structure, '{}/scorecard_structure.pkl'.format(self.ml_res_save_path))
        log.info('Step 8: 持久化模型，分箱点，woe值，评分卡结构======>结束')

        log.info('Step 9: 持久化建模中间结果到excel，方便复盘')
        # corr
        train_woe_corr_df = train_woe[selected_features].corr().reset_index()
        test_woe_corr_df = test_woe[selected_features].corr().reset_index()
        if 'oot' in self.data_type_ar:
            oot_woe_corr_df = oot_woe[selected_features].corr().reset_index()

        # 建模中间结果存储
        # 将模型最终评估结果合并到每一步的评估中

        # final_evaluate_df = merge_rows_one_row_df(evaluate_df, name='lr_', stepname='scorecard_evaluate')
        # final_step_evaluate_log_df = pd.concat([selection_evaluate_log_df, stepwise_evaluate_log_df, final_evaluate_df],
        #                                        axis=0).reset_index(drop=True)
        #
        # excel_utils.df_to_excel(final_step_evaluate_log_df, '9.step_evaluate')  # 每步筛选的评估结果
        # excel_utils.df_to_excel(stepwise_df.reset_index(), '10.model_stepwise')  # stepwise模型结果
        # sw.scorecard_to_excel(scorecard_structure, workbook, 'scorecard_structure')
        excel_utils.df_to_excel(var_info.reset_index().rename(columns={'index': 'var_name'}), '9.var_info')
        excel_utils.df_to_excel(scorecard_structure, 'scorecard_structure')

        in_model_var_train_summary = train_var_summary[train_var_summary['var_name'].isin(selected_features)]
        in_model_var_test_summary = test_var_summary[test_var_summary['var_name'].isin(selected_features)]
        if 'oot' in self.data_type_ar:
            in_model_var_oot_summary = oot_var_summary[oot_var_summary['var_name'].isin(selected_features)]
        autobmt.var_summary_to_excel(in_model_var_train_summary, workbook, '11.in_model_var_train_summary')
        autobmt.var_summary_to_excel(in_model_var_test_summary, workbook, '12.in_model_var_test_summary')
        if 'oot' in self.data_type_ar:
            autobmt.var_summary_to_excel(in_model_var_oot_summary, workbook, '13.in_model_var_oot_summary')

        excel_utils.df_to_excel(train_woe_corr_df, '14.train_woe_df_corr')
        excel_utils.df_to_excel(test_woe_corr_df, '15.test_woe_df_corr')
        if 'oot' in self.data_type_ar:
            excel_utils.df_to_excel(oot_woe_corr_df, '16.oot_woe_df_corr')
        excel_utils.close_workbook()

        if 'oot' in self.data_type_ar:
            autobmt.plot_var_bin_summary(
                {'train': in_model_var_train_summary, 'test': in_model_var_test_summary,
                 'oot': in_model_var_oot_summary},
                selected_features, target=self.target, by=self.data_type_ar, file_path=excel_file_name_path,
                sheet_name='plot_var_bin_summary')
        else:
            autobmt.plot_var_bin_summary(
                {'train': in_model_var_train_summary, 'test': in_model_var_test_summary},
                selected_features, target=self.target, by=self.data_type_ar, file_path=excel_file_name_path,
                sheet_name='plot_var_bin_summary')

        time_end = time.time()
        time_c = time_end - time_start
        log.info('*' * 30 + '建模相关结果保存完成！！！保存路径为：{}'.format(self.ml_res_save_path) + '*' * 30)
        log.info('模型效果：\n{}'.format(lr_auc_ks_psi))
        log.info('time cost {} s'.format(time_c))

        return lr, selected_features

    @classmethod
    def predict(cls, to_pred_df=None, model_path=None):
        if to_pred_df is None:
            raise ValueError('需要进行预测的数据集不能为None，请指定数据集！！！')
        if model_path is None:
            raise ValueError('模型路径不能为None，请指定模型文件路径！！！')
        fb = autobmt.load_from_pkl('{}/fb.pkl'.format(model_path))
        woetf = autobmt.load_from_pkl('{}/woetf.pkl'.format(model_path))
        lrmodel = autobmt.load_from_pkl('{}/lrmodel.pkl'.format(model_path))
        selected_features = autobmt.load_from_pkl('{}/in_model_var.pkl'.format(model_path))

        bin_data = fb.transform(to_pred_df)
        woe_data = woetf.transform(bin_data)
        to_pred_df['p'] = lrmodel.predict_proba(woe_data[selected_features])[:, 1]

        return to_pred_df
