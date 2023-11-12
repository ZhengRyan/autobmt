#!/usr/bin/env python
# ! -*- coding: utf-8 -*-


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd

from autobmt.feature_selection import select_features_by_miss, select_features_by_concentrate, \
    select_features_by_iv_diff, select_features_by_iv, select_features_by_corr, select_features_by_psi

np.random.seed(1)

LENGTH = 500

A = np.random.rand(LENGTH)
A[np.random.choice(LENGTH, 20, replace=False)] = np.nan

B = np.random.randint(100, size=LENGTH)
C = A + np.random.normal(0, 0.2, LENGTH)
D = A + np.random.normal(0, 0.1, LENGTH)

E = np.random.rand(LENGTH)
E[np.random.choice(LENGTH, 480, replace=False)] = np.nan

F = B + np.random.normal(0, 10, LENGTH)

target = np.random.randint(2, size=LENGTH)

frame = pd.DataFrame({
    'A': A,
    'B': B,
    'C': C,
    'D': D,
    'E': E,
    'F': F,
})

noframe = pd.DataFrame({
    'A': np.random.shuffle(A),
    'B': B,
    'C': C,
    'D': D,
    'E': E,
    'F': F,
})

frame['target'] = target
noframe['target'] = target


def test_select_features_by_miss():
    df = select_features_by_miss(frame, threshold=0.8)
    assert 'E' not in df


def test_select_features_by_concentrate():
    df = select_features_by_concentrate(frame, threshold=0.2)
    # assert 'target' not in df


def test_select_features_by_psi():
    df = select_features_by_psi(frame, noframe, threshold=0.05)
    print("select_features_by_psi", df.columns)
    assert 'A' not in df


def test_select_features_by_iv():
    df = select_features_by_iv(frame, target='target', threshold=0.25)
    assert 'B' not in df


def test_select_features_by_iv_diff():
    df = select_features_by_iv_diff(frame, noframe, target='target', threshold=2)
    assert 'A' not in df


def test_select_features_by_corr():
    df = select_features_by_corr(frame, target='target')
    assert ['D', 'E', 'F', 'target'] == df.columns.tolist()

# import time
#
# import numpy as np
# import pandas as pd
#
# cust_id = 'gid'
# target = 'y_label'
# apply_time = 'apply_time'
# data_type = 'type'
#
# time_start = time.time()
#
# exclude_cols = [cust_id, target, data_type]
#
# train_data = pd.read_csv('./ym_train_data.csv')
# oot_data = pd.read_csv('./ym_test_data.csv')
#
# train_data.replace({-999999: np.nan}, inplace=True)
# oot_data.replace({-999999: np.nan}, inplace=True)
#
# train_data['type'] = 'train'
# oot_data['type'] = 'test'
#
# all_data = train_data.append(oot_data)
#
# merge_col = list(train_data.columns)
# merge_col.remove(target)
# merge_col.remove(cust_id)
# merge_col.remove(data_type)
#
# from .transformer import FeatureBin
# from .feature_selection import select_features_by_miss, select_features_by_concentrate, select_features_by_iv, \
#     select_features_by_iv_diff, select_features_by_corr, select_features_by_psi

# ==============================单个单个测试

# # select_features_by_miss
# drop_df, drop_cols = select_features_by_miss(train_data, threshold=0.9, include_cols=merge_col, return_drop=True)
# print(len(drop_cols))
# print("剔除的变量")
# print(drop_cols)
# print(drop_df.shape)
#
# # select_features_by_concentrate
# drop_df, drop_cols = select_features_by_concentrate(train_data, threshold=0.5, include_cols=merge_col, return_drop=True)
# print(len(drop_cols))
# print("剔除的变量")
# print(drop_cols)
# print(drop_df.shape)

# # feature_bin not None
# fb = FeatureBin()
# # select_features_by_iv_diff
# drop_df, drop_cols, iv = select_features_by_iv_diff(train_data, oot_data, target=target, threshold=2, include_cols=merge_col, return_drop=True, return_iv=True, feature_bin=fb)
# print(len(drop_cols))
# print("剔除的变量")
# print(drop_cols)
# iv.to_csv('../tests/feature_selection_test_py_select_features_by_iv_diff____iv_calciv.csv')
# print(drop_df.shape)

# select_features_by_iv
# drop_df, drop_cols, iv = select_features_by_iv(train_data, target=target, threshold=0.02, include_cols=merge_col, return_drop=True, return_iv=True, feature_bin=fb)
# drop_df, drop_cols, iv = select_features_by_iv(train_data, target=target, threshold=0.02, include_cols=merge_col, return_drop=True, return_iv=True)
# print(len(drop_cols))
# print("剔除的变量")
# print(drop_cols)
# #iv.to_csv('../tests/feature_selection_test_py_select_features_by_iv____iv_calciv_check.csv')
# iv.to_csv('../tests/feature_selection_test_py_select_features_by_iv____iv_calciv_check_update.csv')
# print(drop_df.shape)

# # select_features_by_iv_diff
# drop_df, drop_cols, iv = select_features_by_iv_diff(train_data, oot_data, target=target, threshold=2,include_cols=merge_col, return_drop=True, return_iv=True)
# print(len(drop_cols))
# print("剔除的变量")
# print(drop_cols)
# iv.to_csv('../tests/feature_selection_test_py_select_features_by_iv_diff____iv.csv')
# print(drop_df.shape)
#
# # select_features_by_corr
# drop_df, drop_cols, iv = select_features_by_corr(train_data, target=target, threshold=0.7, include_cols=merge_col,return_drop=True)
# print(len(drop_cols))
# print("剔除的变量")
# print(drop_cols)
# iv.to_csv('../tests/feature_selection_test_py_select_features_by_corr____iv.csv')
# print(drop_df.shape)
#
# #select_features_by_psi
# drop_df, drop_cols, psi = select_features_by_psi(train_data, oot_data, target=target, threshold=0.05,include_cols=merge_col, return_drop=True, feature_bin=True)
# print(len(drop_cols))
# print("剔除的变量")
# print(drop_cols)
# psi.to_csv('../tests/feature_select_features_by_psi____psi.csv')
# print(drop_df.shape)

# ==============================单个单个测试

# #=================删除指定变量，测试相关性
# del_c = ['E138', 'D033', 'E092', 'D159', 'E086', 'E137', 'D064', 'D083', 'D058', 'E128', 'D035', 'D165', 'E084', 'D152', 'D008', 'D012', 'D170', 'D049', 'D024', 'D009', 'D084', 'D132', 'D110', 'D025', 'E125', 'E136', 'D034', 'E150', 'D088', 'D108', 'D129', 'D037', 'D102', 'D169', 'D109', 'D061', 'D161', 'E116', 'D174', 'D013', 'E104', 'D010', 'D133', 'D082', 'E101', 'D153', 'D079', 'E122', 'D085', 'E087', 'E102', 'E093', 'D011', 'D078', 'D036', 'E131', 'D059', 'D032', 'D128', 'E106', 'D060', 'D173', 'D103', 'E085', 'D134', 'D130', 'E132']
# # select_features_by_corr
# drop_df, drop_cols, iv = select_features_by_corr(train_data.drop(columns=del_c), target=target, threshold=0.7, include_cols=list(set(merge_col)-set(del_c)),return_drop=True)
# print(len(drop_cols))
# print("剔除的变量")
# print(drop_cols)
# iv.to_csv('../tests/feature_selection_test_py_select_features_by_corr____iv删除特定变量后的.csv')
# print(drop_df.shape)


# #==============================流程测试
# from ryan.feature_selection import FeatureSelection
# #from ryan.feature_selection_v1 import FeatureSelection
# #===psi优先
# # fs_dic = {
# #         "psi": {'threshold': 0.05},
# #         "empty": {'threshold': 0.9},
# #         "const": {'threshold': 0.5},
# #         "iv": {'threshold': 0.02},
# #         "iv_diff": {'threshold': 2, 'target': target},
# #         "corr": {'threshold': 0.7},
# #
# #     }
# #
# # fs_dic = {
# #         "iv": {'threshold': 0.02},
# #         "psi": {'threshold': 0.05},
# #         "empty": {'threshold': 0.9},
# #         "const": {'threshold': 0.5},
# #         "iv_diff": {'threshold': 2, 'target': target},
# #         "corr": {'threshold': 0.7},
# #
# #     }
# #
# # fs_dic = {
# #         "iv_diff": {'threshold': 2, 'target': target},
# #         "empty": {'threshold': 0.9},
# #         "const": {'threshold': 0.5},
# #         "iv": {'threshold': 0.02},
# #         "psi": {'threshold': 0.05},
# #         "corr": {'threshold': 0.7},
# #
# #     }
# #
# # fs_dic = {
# #         "corr": {'threshold': 0.7},
# #         "empty": {'threshold': 0.9},
# #         "const": {'threshold': 0.5},
# #         "iv": {'threshold': 0.02},
# #         "psi": {'threshold': 0.05},
# #         "iv_diff": {'threshold': 2, 'target': target},
# #
# #     }
# #
# # fs_dic = {
# #         "empty": {'threshold': 0.9},
# #         "const": {'threshold': 0.5},
# #         "iv": {'threshold': 0.02},
# #         "psi": {'threshold': 0.05},
# #         "iv_diff": {'threshold': 2, 'target': target},
# #         "corr": {'threshold': 0.7},
# #
# #     }
# #
# # fs_dic = {
# #         "const": {'threshold': 0.5},
# #         "empty": {'threshold': 0.9},
# #         "iv": {'threshold': 0.02},
# #         "psi": {'threshold': 0.05},
# #         "iv_diff": {'threshold': 2, 'target': target},
# #         "corr": {'threshold': 0.7},
# #
# #     }
#
# fs_dic = {
#         "psi": {'threshold': 0.05},
#         "empty": {'threshold': 0.9},
#         "const": {'threshold': 0.5},
#         "corr": {'threshold': 0.7},
#         "iv_diff": {'threshold': 2},
#         "iv": {'threshold': 0.02},
#
#     }
#
#
# print("===============fs_dicfs_dic===============")
#
# fs = FeatureSelection(df=all_data, target=target, exclude_columns=exclude_cols, match_dict=None,
#                       params=fs_dic)
# single_df, selected_features, select_log_df, selection_evaluate_log_df = fs.select()
# print(selected_features)
# print(len(selected_features)) #82
# print(select_log_df.head(3))
#
# #select_log_df.to_csv('../tests/feature_selection_test_psinot2bin_psi优先_selection.csv', index=False)
# #select_log_df.to_csv('../tests/feature_selection_test_psinot2bin_iv优先_selection.csv', index=False)
# #select_log_df.to_csv('../tests/feature_selection_test_psinot2bin_ivdiff优先_selection.csv', index=False)
# #select_log_df.to_csv('../tests/feature_selection_test_psinot2bin_corr优先_selection.csv', index=False)
# #select_log_df.to_csv('../tests/feature_selection_test_psinot2bin_empty优先_selection.csv', index=False)
# #select_log_df.to_csv('../tests/feature_selection_test_psinot2bin_const优先_selection.csv', index=False)
# select_log_df.to_csv('../tests/feature_selection_test_psinot2bin_psi优先ivivdiff_selection.csv', index=False)
#
#
# # #==============================流程测试无指定筛选方式和阀值筛选
# # fs = FeatureSelection(df=all_data, target=target, exclude_columns=exclude_cols, match_dict=None,)
# # single_df, selected_features, select_log_df, selection_evaluate_log_df = fs.select()
# # print(selected_features)
# # print(len(selected_features)) #82
# # print(select_log_df.head(3))
# # select_log_df.to_csv('../tests/feature_selection_test_psinot2bin_无params.csv', index=False)
#
#
#
# # selection_evaluate_log_df.to_csv('selection_evaluate_log_df_new.csv', index=False)
# time_end = time.time()
# time_c = time_end - time_start
# print('time cost {} s'.format(time_c))
# # time cost 6.1279308795928955 s
# ####################################new
# import sys
#
# sys.exit(0)
