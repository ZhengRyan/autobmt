#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: autobmt_lr_tutorial_code.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2022-07-11
'''

import pandas as pd

from autobmt.auto_build_scorecard import AutoBuildScoreCard

##**************************************************虚构现实数据例子**************************************************
##**************************************************虚构现实数据例子**************************************************
##**************************************************虚构现实数据例子**************************************************

##**************************************************自动构建lr模型**************************************************

###TODO 注意修改，读取建模数据
data = pd.read_csv('./example_data/tutorial_data.csv')
data_dict = pd.read_excel('./example_data/tutorial_data数据字典.xlsx')  # 读取数据字典，非必要
###TODO 注意修改，读取建模数据

###TODO 注意修改
client_batch = 'TT01p1'
key, target, data_type = 'APP_ID_C', 'target', 'type'  # key是主键字段名，target是目标变量y的字段名，data_type是train、test数据集标识的字段名
ml_res_save_path = './example_model_result/{}'.format(client_batch)  # 模型结果保存的位置
###TODO 注意修改

###TODO 下面代码基本可以不用动
# 初始化
autobtmodel = AutoBuildScoreCard(datasets=data,  # 训练模型的数据集
                                 fea_names=list(data.columns),  # 数据集的字段名
                                 target=target,  # 目标变量y字段名
                                 key=key,  # 主键字段名
                                 data_type=data_type,  # train、test数据集标识的字段名
                                 no_feature_names=[key, target, data_type] + ['apply_time'],
                                 # 数据集中不用于开发模型的特征字段名，即除了x特征的其它字段名
                                 ml_res_save_path=ml_res_save_path,  # 建模相关结果保存路径
                                 data_dict=data_dict,  # 数据字典，非必要，有则添加，无则不要此参数
                                 AB={'A': 404.65547021957406, 'B': 72.13475204444818},    #自定义的大A，大B；非必要，有默认的
                                 )

# ###训练模型
model, in_model_fea = autobtmodel.fit(empty_threhold=0.95,  # 特征的缺失值大于该阀值的特征会被剔除
                                      iv_threhold=0.02,  # iv小于该阀值的特征会被剔除
                                      corr_threhold=0.7,  # 相关性大于等于该阀值的特征会被剔除
                                      psi_threhold=0.05  # psi大于等于该阀值的特征会被剔除
                                      )
###TODO 上面代码基本可以不用动


# ##**************************************************读取未来需要预测的数据，使用训练好的模型进行预测**************************************************
#
# ###TODO 注意修改，读取未来需要预测的数据
# data = pd.read_csv(
#     '/Users/ryanzheng/PycharmProjects/auto_build_scorecard/examples/example_data/TT01p1_id_y_fea_to_model.csv')
# ###TODO 注意修改，读取未来需要预测的数据
#
# # 未来有新数据过来，使用训练好的模型进行预测
# offline_pred_res = AutoBuildScoreCard.predict(to_pred_df=data,  # 未来需要预测的数据，id+特征即可
#                            model_path='/Users/ryanzheng/PycharmProjects/autoscorecard/examples/example_model_result/TT01p1newcode/20231026180449_1698314689',
#                            # 训练好的模型路径
#                            )
# offline_pred_res.to_csv(
#     '/Users/ryanzheng/PycharmProjects/autoscorecard/examples/example_model_result/TT01p1newcode/20231026180449_1698314689/offline_pred_res.csv',
#     index=False)
#
# ##**************************************************虚构现实数据例子**************************************************
# ##**************************************************虚构现实数据例子**************************************************
# ##**************************************************虚构现实数据例子**************************************************
