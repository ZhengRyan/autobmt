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

from autobmt.metrics import get_auc, get_ks, psi

np.random.seed(1)

feature = np.random.rand(500)
target = np.random.randint(2, size=500)
base_feature = np.random.rand(500)

test_df = pd.DataFrame({
    'A': np.random.rand(500),
    'B': np.random.rand(500),
})
base_df = pd.DataFrame({
    'A': np.random.rand(500),
    'B': np.random.rand(500),
})

FUZZ_THRESHOLD = 1e-10


def test_get_ks():
    result = get_ks(target, feature)
    assert result == 0.055367756612569874


def test_get_auc():
    result = get_auc(target, feature)
    assert result == 0.5038690142424582


def test_PSI():
    result = psi(feature, base_feature, featurebin=[0.3, 0.5, 0.7])
    assert result == 0.018630024627491467


def test_PSI_frame():
    result = psi(
        test_df,
        base_df,
        featurebin={
            'A': [0.3, 0.5, 0.7],
            'B': [0.4, 0.8],
        },
    )

    assert result['B'] == pytest.approx(0.014528279995858708, FUZZ_THRESHOLD)


def test_PSI_return_frame():
    result, frame = psi(
        test_df,
        base_df,
        featurebin={
            'A': [0.3, 0.5, 0.7],
            'B': [0.4, 0.8],
        },
        return_frame=True,
    )

    assert frame.loc[4, 'no_base'] == 0.38
