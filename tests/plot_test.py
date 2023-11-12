#!/usr/bin/env python
# ! -*- coding: utf-8 -*-


import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd

from autobmt.plot import plot_var_bin_summary
from autobmt.transformer import FeatureBin, WoeTransformer, DEFAULT_NAME

np.random.seed(1)

ab = np.array(list('ABCDEFG'))
feature = np.random.randint(10, size=500)
target = np.random.randint(2, size=500)
str_feat = ab[np.random.choice(7, 500)]
uni_feat = np.ones(500)
empty_feat = feature.astype(float)
empty_feat[np.random.choice(500, 50, replace=False)] = np.nan
ty = np.array(['train', 'test', 'oot'])
ty_fea = ty[np.random.choice(3, 500)]

df = pd.DataFrame({
    'A': feature,
    'B': str_feat,
    'C': uni_feat,
    'D': empty_feat,
    'target': target,
    'type': ty_fea,
})


def test_plot_var_bin_summary():
    fb = FeatureBin()
    fb.fit(df.drop(['type'], axis=1), 'target', n_bins=4, is_empty_bin=True, is_need_monotonic=False)
    df_bin = fb.transform(df, labels=True)

    assert df_bin.loc[495, 'B'] == '0.G,A,D'

    file_path = './tests/polt_bin_summary/'
    cols = ['A', 'B', 'C', 'D', 'type']
    plot_var_bin_summary(df_bin, cols, by='type', file_path=file_path, sheet_name='Sheet', need_bin=False)


fb = FeatureBin()
fb.fit(df.drop(['type'], axis=1), 'target', n_bins=4, is_empty_bin=True, is_need_monotonic=False)
df_bin = fb.transform(df, labels=True)

assert df_bin.loc[495, 'B'] == '0.G,A,D'

# file_path = '../tests/polt_bin_summaryttttTruedtatatat/'
# cols = ['A', 'B', 'C', 'D', 'type']
# plot_var_bin_summary(df, cols, by='type', file_path=file_path, sheet_name='Sheet',need_bin=True,is_need_monotonic=True,method='dt')
