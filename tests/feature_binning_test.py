#!/usr/bin/env python
# ! -*- coding: utf-8 -*-


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
from autobmt.feature_binning import bin_method_run, chi_bin, dt_bin, equal_freq_bin, kmeans_bin
#from .feature_binning import bin_method_run, chi_bin, dt_bin, equal_freq_bin, kmeans_bin

np.random.seed(1)
feature = np.random.rand(500)
t = np.random.randint(2, size=500)
Aa = np.random.randint(100, size=500)
Bb = np.random.randint(3, size = 500)

df = pd.DataFrame({
    'feature': feature,
    'target': t,
    'A': Aa,
})

LENGTH = 500
np.random.seed(1)
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

frame['target'] = target


def test_chimerge():
    splits = chi_bin(feature, t, n_bins=10, is_need_monotonic=False)
    print(splits)
    assert len(splits) == 9


def test_chimerge_bins_not_enough():
    splits = chi_bin(Bb, t, n_bins=10, is_need_monotonic=False)
    assert len(splits) == 2


def test_chimerge_bins_with_min_samples():
    splits = chi_bin(feature, t, n_bins=15, min_sample_rate=0.02, is_need_monotonic=False)
    print(len(splits))
    assert len(splits) == 10


def test_dtmerge():
    splits = dt_bin(feature, t, n_bins=10, is_need_monotonic=False)
    assert len(splits) == 9


def test_quantilemerge():
    splits = equal_freq_bin(feature, n_bins=10, is_need_monotonic=False)
    assert len(splits) == 9


def test_quantilemerge_not_enough():
    splits = equal_freq_bin(Bb, n_bins=10, is_need_monotonic=False)
    assert len(splits) == 2


# def test_stepmerge():
#     splits = StepMerge(feature, n_bins = 10)
#     assert len(splits) == 9

def test_kmeansmerge():
    splits = kmeans_bin(feature, n_bins=10, is_need_monotonic=False)
    assert len(splits) == 9


def test_merge():
    splits, res = bin_method_run(feature, target=t, method='chi', return_bin=True, n_bins=10,
                                 is_need_monotonic=False)
    assert len(np.unique(res)) == 10


def test_merge_frame():
    res = bin_method_run(df, target='target', method='chi', return_bin=True, n_bins=10, is_need_monotonic=False)
    assert len(np.unique(res['feature'])) == 10


def test_merge_frame2():
    res = bin_method_run(df, target='target', method='chi', n_bins=10, is_need_monotonic=False)
    assert len(np.unique(res['feature'])) == 10






print("======================测试equal_freq_bin")
def test_equal_freq_bin1():
    s = equal_freq_bin(D, target, n_bins=21, is_need_monotonic=False)
    print("=======len(s)======", len(s))
    assert list(s)[:-1] == list(
        [0.012419821725897555, 0.08160251414710648, 0.1365937204837973, 0.20126781529042354, 0.26078161584175596,
         0.3043923889835296, 0.37010564491684994, 0.43393496112637275, 0.4761375755670674, 0.5346413336617297,
         0.5827083140162775, 0.623464641428682, 0.6750798738210625, 0.7170075229286542, 0.77298483315698,
         0.8173532653547075, 0.8517076655269138, 0.9129958438409241, 0.9850361692249885])

def test_equal_freq_bin2():
    s = equal_freq_bin(D, target, n_bins=10, is_need_monotonic=False)
    assert len(s) == 10

def test_equal_freq_bin3():
    s = equal_freq_bin(D, target, n_bins=21)
    assert list(s)[:-1] == list([0.5346413336617297])

def test_equal_freq_bin4():
    ##测默认箱子
    s = equal_freq_bin(D, target, is_need_monotonic=False)
    assert len(s) == 9

def test_equal_freq_bin5():
    s = bin_method_run(frame[['D', 'target']], target='target', method='equal_freq', n_bins=21, is_need_monotonic=True)
    assert len(np.unique(s['D'])) == 3

def test_equal_freq_bin6():
    s = bin_method_run(frame[['D', 'target']], target='target', method='equal_freq', n_bins=21)
    assert len(np.unique(s['D'])) == 3

def test_equal_freq_bin7():
    s = bin_method_run(frame[['D', 'target']], target='target', method='equal_freq', n_bins=21, is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 21

def test_equal_freq_bin8():
    s = bin_method_run(frame[['D', 'target']], target='target', method='equal_freq')
    assert len(np.unique(s['D'])) == 3

def test_equal_freq_bin9():
    s = bin_method_run(frame[['D', 'target']], target='target', method='equal_freq', n_bins=21)
    assert len(np.unique(s['D'])) == 3

def test_equal_freq_bin10():
    s = bin_method_run(frame[['D', 'target']], target, method='equal_freq', n_bins=21, is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 21

def test_equal_freq_bin11():
    s = bin_method_run(frame[['D', 'target']], target=None, method='equal_freq', n_bins=21, is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 21

def test_equal_freq_bin12():
    s = bin_method_run(frame[['D', 'target']], target=None, method='equal_freq', n_bins=21)
    assert len(np.unique(s['D'])) == 21

def test_equal_freq_bin13():
    ##测默认箱子
    s = bin_method_run(frame[['D', 'target']], target, method='equal_freq', is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 10

def test_equal_freq_bin14():
    s = bin_method_run(frame[['D', 'target']].fillna(-9), target, method='equal_freq', is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 10

print("======================测试equal_freq_bin")
print("======================测试kmeans")
def test_kmeans_bin1():
    s = kmeans_bin(D, target, n_bins=21, is_need_monotonic=False)
    print("=======len(s)======", len(s))
    assert list(s)[:-1] == list(
        [0.05394007952037447, 0.1886799394187435, 0.31125728976809286, 0.4206303104301733, 0.5315341201997051,
         0.6423777917886719, 0.7568052721344689, 0.8707566397289682, 0.9905607587443476])

def test_kmeans_bin2():
    s = kmeans_bin(D, target, n_bins=10, is_need_monotonic=False)
    print("=======len(s)======", len(s))
    assert len(s) == 10

def test_kmeans_bin3():
    s = kmeans_bin(D, target, n_bins=21)
    assert list(s)[:-1] == list([0.491896259821787])

def test_kmeans_bin4():
    ##测默认箱子
    s = kmeans_bin(D, target, is_need_monotonic=False)
    assert len(s) == 9

def test_kmeans_bin5():
    s = bin_method_run(frame[['D', 'target']], target='target', method='kmeans', n_bins=21, is_need_monotonic=True)
    assert len(np.unique(s['D'])) == 3

def test_kmeans_bin6():
    s = bin_method_run(frame[['D', 'target']], target='target', method='kmeans', n_bins=21)
    assert len(np.unique(s['D'])) == 3

def test_kmeans_bin7():
    s = bin_method_run(frame[['D', 'target']], target='target', method='kmeans', n_bins=21, is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 11

def test_kmeans_bin8():
    s = bin_method_run(frame[['D', 'target']], target='target', method='kmeans')
    assert len(np.unique(s['D'])) == 3

def test_kmeans_bin9():
    s = bin_method_run(frame[['D', 'target']], target='target', method='kmeans', n_bins=21)
    assert len(np.unique(s['D'])) == 3

def test_kmeans_bin10():
    s = bin_method_run(frame[['D', 'target']], target, method='kmeans', n_bins=21, is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 11

def test_kmeans_bin11():
    s = bin_method_run(frame[['D', 'target']], target=None, method='kmeans', n_bins=21, is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 11

def test_kmeans_bin12():
    s = bin_method_run(frame[['D', 'target']], target=None, method='kmeans', n_bins=21)
    assert len(np.unique(s['D'])) == 11

def test_kmeans_bin13():
    ##测默认箱子
    s = bin_method_run(frame[['D', 'target']], target, method='kmeans', is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 10

def test_kmeans_bin14():
    s = bin_method_run(frame[['B', 'target']], target, method='kmeans', is_need_monotonic=False)
    assert len(np.unique(s['B'])) == 10
print("======================测试kmeans")
print("======================测试dt")

def test_dt_bin1():
    s = dt_bin(D, target, n_bins=21, is_need_monotonic=False)
    print("=======len(s)======", len(s))
    assert list(s)[:-1] == list(
        [0.03355235606431961, 0.14810966700315475, 0.2149350941181183, 0.3048485219478607, 0.37758709490299225,
         0.44229236245155334, 0.49767398834228516, 0.5686621367931366, 0.6086502075195312, 0.6941415071487427,
         0.7509821057319641, 0.8093653619289398, 0.843919038772583, 0.899420440196991, 0.9890741109848022])

def test_dt_bin2():
    s = dt_bin(D, target, n_bins=10, is_need_monotonic=False)
    assert len(s) == 10

def test_dt_bin3():
    s = dt_bin(D, target, n_bins=21)
    assert list(s)[:-1] == list([0.9890741109848022])

def test_dt_bin4():
    ##测默认箱子
    s = dt_bin(D, target, is_need_monotonic=False)
    assert len(s) == 9

def test_dt_bin5():
    s = bin_method_run(frame[['D', 'target']], target='target', method='dt', n_bins=21, is_need_monotonic=True)
    assert len(np.unique(s['D'])) == 3

def test_dt_bin6():
    s = bin_method_run(frame[['D', 'target']], target='target', method='dt', n_bins=21)
    assert len(np.unique(s['D'])) == 3

def test_dt_bin7():
    s = bin_method_run(frame[['D', 'target']], target='target', method='dt', n_bins=21, is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 17

def test_dt_bin8():
    s = bin_method_run(frame[['D', 'target']], target='target', method='dt')
    assert len(np.unique(s['D'])) == 3

def test_dt_bin9():
    s = bin_method_run(frame[['D', 'target']], target, method='dt', n_bins=21, is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 17

def test_dt_bin10():
    ##测默认箱子
    s = bin_method_run(frame[['D', 'target']], target, method='dt', is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 10

def test_dt_bin11():
    s = bin_method_run(frame[['D', 'target']].fillna(-9), target, method='dt', is_need_monotonic=False)
    assert len(np.unique(s['D'])) == 10
print("======================测试dt")
print("======================测试chi")
def test_chi_bin1():
    s = chi_bin(A, target, n_bins=21, is_need_monotonic=False)
    print("=======len(s)======", len(s))
    assert list(s)[:-1] == list([0.1354279030721699, 0.4108113499221856, 0.7438258540750929, 0.8763891522960383])

def test_chi_bin2():
    s = chi_bin(A, target, n_bins=4, is_need_monotonic=False)
    assert len(s) == 4

def test_chi_bin3():
    s = chi_bin(A, target, n_bins=13, is_need_monotonic=False)
    assert len(s) == 5

def test_chi_bin4():
    s = chi_bin(A, target, n_bins=21)
    print("=======len(s)======", len(s))
    assert list(s)[:-1] == list([0.7438258540750929])

def test_chi_bin5():
    ##测默认箱子
    s = chi_bin(A, target, is_need_monotonic=False)
    print("=======len(s)======", len(s))
    assert len(s) == 5

def test_chi_bin6():
    s = bin_method_run(frame[['A', 'target']], target='target', method='chi', n_bins=21, is_need_monotonic=True)
    assert len(np.unique(s['A'])) == 3

def test_chi_bin7():
    s = bin_method_run(frame[['A', 'target']], target='target', method='chi', n_bins=21)
    assert len(np.unique(s['A'])) == 3

def test_chi_bin8():
    s = bin_method_run(frame[['A', 'target']], target='target', method='chi', n_bins=21, is_need_monotonic=False)
    assert len(np.unique(s['A'])) == 6

def test_chi_bin9():
    s = bin_method_run(frame[['A', 'target']], target='target', method='chi')
    assert len(np.unique(s['A'])) == 3

def test_chi_bin10():
    s = bin_method_run(frame[['A', 'target']], target, method='chi', n_bins=21, is_need_monotonic=False)
    assert len(np.unique(s['A'])) == 6

def test_chi_bin11():
    ##测默认箱子
    s = bin_method_run(frame[['A', 'target']], target, method='chi', is_need_monotonic=False)
    assert len(np.unique(s['A'])) == 6

def test_chi_bin12():
    s = bin_method_run(frame[['A', 'target']].fillna(-9), target, method='chi', is_need_monotonic=False)
    assert len(np.unique(s['A'])) == 5
print("======================测试chi")




# #####测试决策树分箱
# def split_empty(feature, y=None, return_mask=True):
#     copy_fea = np.copy(feature)
#     mask = pd.isna(copy_fea)
#
#     copy_fea = copy_fea[~mask]
#     if y is not None:
#         copy_y = np.copy(y)
#         copy_y = copy_y[~mask]
#         return copy_fea, copy_y, mask
#     return copy_fea, mask
#
# from sklearn.tree import DecisionTreeClassifier, _tree
# tree = DecisionTreeClassifier(
#     min_samples_leaf=1,
#     max_leaf_nodes=22,
#     # 优先满足min_samples_leaf参数。在满足min_samples_leaf参数参数后，再考虑max_leaf_nodes。
#     # 比如情况1：min_samples_leaf设置成0.05，max_leaf_nodes设置成20。满足0.05后，最大max_leaf_nodes只有10，那也就这样了
#     # 比如情况2：min_samples_leaf设置成0.05，max_leaf_nodes设置成6。满足0.05后，最大max_leaf_nodes有10，那再考虑max_leaf_nodes，继续分到满足max_leaf_nodes=6停止
#     # ps:min_samples_leaf=1表示没有限制
# )
#
# feature, target, empty_mask = split_empty(D, target)
#
# tree.fit(feature.reshape((-1, 1)), target)
# thresholds = tree.tree_.threshold
# thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]
# splits = np.sort(thresholds)
# print(len(splits))
# #####测试决策树分箱
