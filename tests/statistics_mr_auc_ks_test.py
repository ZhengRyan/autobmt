#!/usr/bin/env python
#! -*- coding: utf-8 -*-

'''
@File: statistics_mr_auc_ks_test.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2023-12-06
'''

from autobmt.statistics import StatisticsMrAucKs

conf_dict = {
        'cust_id': 'device_id',  # 主键
        'date_na': 'apply_time',  # 时间列名
        'type_na': 'type',  # 数据集列名
        'model_pred_res': 'p',  # 模型预测值
        'target_na': 'target',  # 目标变量列名
        'device_type_na': 'device_type',  # 设备列名
        'year_month_na': 'apply_month',

        ###设备号列名
        'oaid_col_na': 'oaid_md5',
        'imei_col_na': 'imei_md5',
        'idfa_col_na': 'idfa_md5',

        ###设备号的具体取值
        'oaid_value': 'oaid',
        'imei_value': 'imei',
        'idfa_value': 'idfa',
    }

data_path = '/Users/ryanzheng/td/项目/RTA/拍拍贷RTA/模型/R8431/TD84p31_oneid_v2_all_rowdrop_to_statistics_mr_auc_ks.csv'
smak = StatisticsMrAucKs(conf_dict, data_path)
smak.statistics_model_mr_auc_ks()

