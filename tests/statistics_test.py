#!/usr/bin/env python
# ! -*- coding: utf-8 -*-


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import pytest

from autobmt.statistics import WOE, calc_iv, _IV, get_vif, calc_var_summary, get_iv_psi, calc_bin_summary

np.random.seed(1)

feature = np.random.rand(500)
target = np.random.randint(2, size=500)
A = np.random.randint(100, size=500)
B = np.random.randint(100, size=500)
mask = np.random.randint(8, size=500)

df = pd.DataFrame({
    'feature': feature,
    'target': target,
    'A': A,
    'B': B,
})


def test_woe():
    value = WOE(0.2, 0.3)
    assert value == pytest.approx(-0.4054651081081643)


def test_iv_priv():
    value = _IV(df['feature'], df['target'])
    assert value == pytest.approx(0.010385942643745403)


def test_iv():
    value = calc_iv(df['feature'], df['target'], n_bins=10, method='dt')
    assert value == pytest.approx(0.23561308290317584)


def test_iv_frame():
    res = calc_iv(df, 'target', n_bins=10, method='chi')
    assert res.loc[0, 'A'] == pytest.approx(0.04517306679551927)


def test_vif():
    vif = get_vif(df)
    assert vif['A'] == 2.9515204239842703


def test_calc_var_summary():
    var_s = calc_var_summary(df, target='target')
    assert list(var_s.loc[4, 'IV'].values) == [0.036723641370124835, 0.0605206894133002, 0.07822121543008585]


def test_get_iv_psi():
    df['type'] = 'train'
    var_iv_psi = get_iv_psi(df, feature_list=['feature', 'A', 'B'], by_col='type', only_psi=False)
    assert list(var_iv_psi.IV) == [0.036723641370124835, 0.0605206894133002, 0.07822121543008587]


dff = pd.DataFrame({
    'feature': feature,
    'target': target,
})


def test_calc_bin_summary():
    result = calc_bin_summary(dff, bin_col='feature', target='target')
    assert result.loc[4, 'ks'] == -0.028036335090277087


# def test_calc_bin_summary_use_step():
#     result = calc_bin_summary(df,bin_col='feature', target='target', method = 'step', clip_q = 0.01)
#     assert result.loc[4, 'ks'] == -0.0422147102645028

def test_calc_bin_summary_for_all_score():
    result = calc_bin_summary(dff, bin_col='feature', target='target', bin=False)
    assert len(result) == 500


def test_calc_bin_summary_return_splits():
    result = calc_bin_summary(dff, bin_col='feature', target='target')
    assert len(result) == 10


def test_calc_bin_summary_use_split_pointers():
    result = calc_bin_summary(dff, bin_col='feature', target='target', bin=[0.2, 0.6])
    assert len(result) == 3


def test_calc_bin_summary_with_lift():
    result = calc_bin_summary(dff, bin_col='feature', target='target', is_sort=False)
    assert result.loc[3, 'lift'] == 1.0038610038610039


def test_calc_bin_summary_with_cum_lift():
    result = calc_bin_summary(dff, bin_col='feature', target='target', is_sort=False)
    assert result.loc[3, 'cum_lift'] == 1.003861003861004
