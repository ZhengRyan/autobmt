#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: autobmt_treemodel_tutorial_code_LGB.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2022-07-11
'''

import pandas as pd

from autobmt.auto_build_tree_model_lgb import AutoBuildTreeModelLGB

##**************************************************虚构现实数据例子**************************************************
##**************************************************虚构现实数据例子**************************************************
##**************************************************虚构现实数据例子**************************************************

##**************************************************自动构建lr模型**************************************************

###TODO 注意修改，读取建模数据
data = pd.read_csv('./example_data/tutorial_data.csv')
data_dict = pd.read_excel('./example_data/tutorial_data数据字典.xlsx')  # 读取数据字典，非必要
###TODO 注意修改，读取建模数据

###TODO 注意修改
client_batch = 'TT00p2'
key, target, data_type = 'APP_ID_C', 'target', 'type'  # key是主键字段名，target是目标变量y的字段名，data_type是train、test数据集标识的字段名
ml_res_save_path = './example_model_result/{}'.format(client_batch)  # 模型结果保存的位置
###TODO 注意修改

###TODO 下面代码基本可以不用动
# 初始化
autobtmodel = AutoBuildTreeModelLGB(datasets=data,  # 训练模型的数据集
                                 fea_names=list(data.columns),  # 数据集的字段名
                                 target=target,  # 目标变量y字段名
                                 key=key,  # 主键字段名
                                 data_type=data_type,  # train、test数据集标识的字段名
                                 no_feature_names=[key, target, data_type] + ['apply_time', 'var_d1', 'var_d3',
                                                                              'var_d4', 'var_d5', 'var_d6', 'var_d7',
                                                                              'var_d8', 'var_d11'],
                                 # 数据集中不用于开发模型的特征字段名，即除了x特征的其它字段名
                                 ml_res_save_path=ml_res_save_path,  # 建模相关结果保存路径
                                 AB={'A': 404.65547021957406, 'B': 72.13475204444818},  # 自定义的大A，大B；非必要，有默认的
                                 positive_corr=False,  # 数与模型预测的概率值是否正相关。默认False，负相关，即概率约高，分数越低
                                 )

# 训练模型
model, in_model_fea = autobtmodel.fit(is_feature_select=True,  # 特征筛选
                                      is_auto_tune_params=True,  # 是否自动调参
                                      is_stepwise_del_feature=True,  # 是进行逐步的变量删除
                                      feature_select_method='shap',  # 特征筛选指标
                                      method_threhold=0.001,  # 特征筛选指标阈值
                                      corr_threhold=0.8,  # 相关系数阈值
                                      psi_threhold=0.1,  # PSI阈值
                                      )
###TODO 上面代码基本可以不用动


# ##**************************************************读取未来需要预测的数据，使用训练好的模型进行预测**************************************************
#
# ###TODO 注意修改，读取未来需要预测的数据
# data = pd.read_csv(
#     './example_data/tutorial_data.csv')
# ###TODO 注意修改，读取未来需要预测的数据
#
# # 未来有新数据过来，使用训练好的模型进行预测
# offline_pred_res = AutoBuildTreeModelLGB.predict(to_pred_df=data,  # 未来需要预测的数据，id+特征即可
#                                               model_path='./example_model_result/TT00p2/20231119105732_32',
#                                               # 训练好的模型路径
#                                               )
# offline_pred_res.to_csv(
#     './example_model_result/TT00p2/20231119105732_32/offline_pred_res.csv',
#     index=False)
#
# ##**************************************************虚构现实数据例子**************************************************
# ##**************************************************虚构现实数据例子**************************************************
# ##**************************************************虚构现实数据例子**************************************************
