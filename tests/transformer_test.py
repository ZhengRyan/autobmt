#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pytest
import numpy as np
import pandas as pd

from autobmt.transformer import FeatureBin, WoeTransformer, DEFAULT_NAME

np.random.seed(1)

ab = np.array(list('ABCDEFG'))
feature = np.random.randint(10, size=500)
target = np.random.randint(2, size=500)
str_feat = ab[np.random.choice(7, 500)]
uni_feat = np.ones(500)
empty_feat = feature.astype(float)
empty_feat[np.random.choice(500, 50, replace=False)] = np.nan

df = pd.DataFrame({
    'A': feature,
    'B': str_feat,
    'C': uni_feat,
    'D': empty_feat,
    'target': target,
})

df.to_csv('/Users/ryanzheng/PycharmProjects/autoscorecard/tests/genrate_data.csv', index=False)


def test_duplicated_keys():
    dup_df = df.rename(columns={"C": "A"})
    with pytest.raises(Exception, match=r"X has duplicate keys `.*`"):
        WoeTransformer().fit_transform(dup_df, target)


def test_woe_transformer():
    f = WoeTransformer().fit_transform(feature, target)
    assert f[451] == pytest.approx(-0.17061154127869285)


def test_woe_transformer_with_str():
    f = WoeTransformer().fit_transform(str_feat, target)
    assert f[451] == pytest.approx(-0.2198594761130199)


def test_woe_transformer_with_unknown_group():
    transer = WoeTransformer().fit(str_feat, target)
    res = transer.transform(['Z'], default='min')
    assert res[0] == pytest.approx(-0.2198594761130199)


def test_woe_transformer_frame():
    res = WoeTransformer().fit_transform(df, target)
    assert res.iloc[451, 1] == pytest.approx(-0.2198594761130199)


def test_woe_transformer_dict():
    transer = WoeTransformer().fit(df, 'target')
    res = transer.transform({
        "A": 6,
        "B": "C",
        "C": 1,
        "D": 2,
    })
    assert res['B'].item() == pytest.approx(-0.09149433112609942)


def test_woe_transformer_select_dtypes():
    res = WoeTransformer().fit_transform(df, target, select_dtypes='object')
    assert res.loc[451, 'A'] == 3


def test_woe_transformer_exclude():
    res = WoeTransformer().fit_transform(df, target, exclude='A')
    assert res.loc[451, 'A'] == 3


def test_woe_transformer_export_single():
    transer = WoeTransformer().fit(feature, target)
    t = transer.export()
    assert t[DEFAULT_NAME][5] == pytest.approx(0.3938235330926786)


def test_woe_transformer_export():
    transer = WoeTransformer().fit(df, target)
    t = transer.export()
    assert t['C'][1] == 0


def test_woe_transformer_load():
    rules = {
        'A': {
            1: 0.1,
            2: 0.2,
            3: 0.3,
        }
    }

    transer = WoeTransformer().load(rules)
    assert transer.fea_woe_dict['A'][2] == 0.2


def test_featurebin():
    f = FeatureBin().fit_transform(feature, target, method='chi', is_need_monotonic=False)
    assert f[451] == 3


def test_featurebin_with_str():
    f = FeatureBin().fit_transform(str_feat, target, method='chi', is_need_monotonic=False)
    assert f[451] == 0


def test_featurebin_unique_feature():
    f = FeatureBin().fit_transform(uni_feat, target, method='chi', is_need_monotonic=False)
    assert f[451] == 0


def test_featurebin_frame():
    res = FeatureBin().fit_transform(df, target)
    assert res.iloc[404, 1] == 2


def test_featurebin_select_dtypes():
    res = FeatureBin().fit_transform(df, target, select_dtypes='number')
    assert res.loc[451, 'B'] == 'G'


def test_featurebin_exclude():
    res = FeatureBin().fit_transform(df, target, exclude='B')
    assert res.loc[451, 'B'] == 'G'


def test_featurebin_labels():
    fb = FeatureBin().fit(df, target, is_need_monotonic=False)
    res = fb.transform(df, labels=True)
    assert res.loc[451, 'A'] == '3.[3 ~ 4)'


# def test_featurebin_single_feature():
#     fb = FeatureBin().fit(df['A'], method = 'step', n_bins = 5)
#     res = fb.transform(df['A'])
#     assert res[451] == 1

def test_featurebin_export():
    fb = FeatureBin().fit(df, target, method='chi', n_bins=4, is_need_monotonic=False)
    bins = fb.export(bin_format=False)
    assert isinstance(bins['B'][0], list)


def test_featurebin_update():
    fb = FeatureBin().fit(df, target, method='chi', n_bins=4, is_need_monotonic=False)
    fb.manual_bin({'A': [1, 2, 3, 4, 5, 6]})
    bins = fb.export(bin_format=False)
    assert len(bins['A']) == 6


# def test_featurebin_step():
#     fb = FeatureBin().fit(df['A'], method = 'step', n_bins = 4)
#     bins = fb.export()
#     assert bins['A'][1] == 4.5

def test_featurebin_target_in_frame():
    fb = FeatureBin().fit(df, 'target', n_bins=4, is_need_monotonic=False)
    bins = fb.export(bin_format=False)
    assert bins['A'][1] == 6


def test_featurebin_target_in_frame_kwargs():
    fb = FeatureBin().fit(df, y='target', n_bins=4, is_need_monotonic=False)
    bins = fb.export(bin_format=False)
    assert bins['A'][1] == 6


def test_featurebin_empty_separate():
    fb = FeatureBin()
    bins = fb.fit_transform(df, 'target', n_bins=4, is_empty_bin=True, is_need_monotonic=False)
    mask = pd.isna(df['D'])
    assert (bins['D'][~mask] != 4).all()


def test_featurebin_labels_with_empty():
    fb = FeatureBin().fit(df, 'target', n_bins=4, is_empty_bin=True, is_need_monotonic=False)
    res = fb.transform(df, labels=True)
    assert res.loc[2, 'D'] == '4.nan'
