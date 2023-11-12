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
from sklearn.linear_model import LogisticRegression

from autobmt.scorecard import ScoreCard, FeatureBin, WoeTransformer

np.random.seed(1)

# Create a testing dataframe and a scorecard model.

ab = np.array(list('ABCDEFG'))
feature = np.random.randint(10, size=500)
target = np.random.randint(2, size=500)
str_feat = ab[np.random.choice(7, 500)]

df = pd.DataFrame({
    'A': feature,
    'B': str_feat,
    'C': ab[np.random.choice(2, 500)],
    'D': np.ones(500),
})

df.to_csv('/Users/ryanzheng/PycharmProjects/autoscorecard/tests/genrate_data.csv', index=False)

card_config = {
    'A': {
        '[-inf ~ 3)': 100,
        '[3 ~ 5)': 200,
        '[5 ~ 8)': 300,
        '[8 ~ inf)': 400,
        'nan': 500,
    },
    'B': {
        ','.join(list('ABCD')): 200,
        ','.join(list('EF')): 400,
        'else': 500,
    },
    'C': {
        'A': 200,
        'B': 100,
    },
}

# card_config = {
# 	'A': {
# 		'[-inf ~ 2)': [142.27, 0.053086897685008266, 0.8196986054553417],
# 		'[2 ~ 5)': [151.36, -0.0750378915742799, 0.8196986054553417],
# 		'[5 ~ 6)': [118.09, 0.3938235330926786, 0.8196986054553417],
# 		'[6 ~ 7)': [178.21, -0.453474327294525, 0.8196986054553417],
# 		'[7 ~ inf)': [140.09, 0.08376005844476284, 0.8196986054553417]
# 	},
# 	'B': {
# 		'G,A': [162.08, -0.21634453400557532, 0.8567329286858805],
# 		'D': [159.24, -0.17806234743455837, 0.8567329286858805],
# 		'C': [152.82, -0.09149433112609942, 0.8567329286858805],
# 		'F': [134.04, 0.16171131179570847, 0.8567329286858805],
# 		'B,E': [123.68, 0.30136642227076027, 0.8567329286858805]
# 	},
# 	'C': {
# 		'A': [148.83, -0.07854594304644223, 0.4109435633463568],
# 		'B': [142.95, 0.08664766595989656, 0.4109435633463568]
# 	}
# }


combiner = FeatureBin()
bins = combiner.fit_transform(df, target, n_bins=5, is_need_monotonic=False)
woe_transer = WoeTransformer()
woe = woe_transer.fit_transform(bins, target)

lr = LogisticRegression()
lr.fit(woe, target)

# create a score card
card = ScoreCard(
    combiner=combiner,
    transer=woe_transer,
)

card.fit(woe, target)

FUZZ_THRESHOLD = 1e-6
TEST_SCORE = pytest.approx(414.0772336534059, FUZZ_THRESHOLD)


def test_representation():
    repr(card)


def test_load():
    card = ScoreCard().load(card_config)
    score = card.predict(df)
    assert score[200] == 600


def test_load_after_init_combiner():
    card = ScoreCard(
        combiner=combiner,
        transer=woe_transer,
    )
    card.load(card_config)
    score = card.predict(df)
    assert score[200] == 600


def test_proba_to_score():
    model = LogisticRegression()
    model.fit(woe, target)

    proba = model.predict_proba(woe)[:, 1]
    score = card.proba_to_score(proba)
    assert score[404] == TEST_SCORE


def test_score_to_prob():
    score = card.predict(df)
    proba = card.score_to_proba(score)
    assert proba[404] == pytest.approx(0.4673929989138551, FUZZ_THRESHOLD)


def test_predict():
    score = card.predict(df)
    assert score[404] == TEST_SCORE


def test_predict_proba():
    proba = card.predict_proba(df)
    assert proba[404, 1] == pytest.approx(0.4673929989138551, FUZZ_THRESHOLD)


def test_predict_sub_score():
    score, sub = card.predict(df, return_sub=True)
    assert sub.loc[250, 'B'] == pytest.approx(147.10499540339703, FUZZ_THRESHOLD)


def test_woe_to_score():
    score = card.woe_to_score(woe)
    score = np.sum(score, axis=1)
    assert score[404] == TEST_SCORE


def test_woe_sum_to_score():
    score = card.woe_sum_to_score(woe)
    assert score[404] == TEST_SCORE


def test_bin_to_score():
    score = card.bin_to_score(bins)
    assert score[404] == TEST_SCORE


def test_export_map():
    card_map = card.export()
    assert card_map['B']['D'][0] == 144.74


def test_card_map():
    config = card.export()
    card_from_map = ScoreCard().load(config)
    score = card_from_map.predict(df)
    assert score[404] == 414.0799999999999


def test_card_map_with_else():
    card_from_map = ScoreCard().load(card_config)
    score = card_from_map.predict(df)
    assert score[80] == 1000


def test_export_frame():
    card = ScoreCard().load(card_config)
    frame = card.export(to_dataframe=True)
    rows = frame[(frame['feature'] == 'B') & (frame['value'] == 'else')].reset_index()
    assert rows.loc[0, 'score'] == 500


def test_card_combiner_number_not_match():
    c = combiner.export(bin_format=False)
    c['A'] = [0, 3, 6, 8]
    com = FeatureBin().manual_bin(c)
    bins = com.transform(df)
    woe_transer = WoeTransformer()
    woe = woe_transer.fit_transform(bins, target)

    card = ScoreCard(
        combiner=com,
        transer=woe_transer,
    )

    with pytest.raises(Exception) as e:
        # will raise an exception when fitting a card
        card.fit(woe, target)

    assert '\'A\' is not matched' in str(e.value)


def test_card_combiner_str_not_match():
    c = combiner.export(bin_format=False)
    c['C'] = [['A'], ['B'], ['C']]
    com = FeatureBin().manual_bin(c)
    bins = com.transform(df)
    woe_transer = WoeTransformer()
    woe = woe_transer.fit_transform(bins, target)

    card = ScoreCard(
        combiner=com,
        transer=woe_transer,
    )

    with pytest.raises(Exception) as e:
        # will raise an exception when fitting a card
        card.fit(woe, target)

    assert '\'C\' is not matched' in str(e.value)


def test_card_with_less_X():
    x = woe.drop(columns='A')
    card = ScoreCard(
        combiner=combiner,
        transer=woe_transer,
    )

    card.fit(x, target)
    assert card.predict(df)[200] == pytest.approx(417.43225487849736, FUZZ_THRESHOLD)


def test_card_predict_with_unknown_feature():
    np.random.seed(9)
    unknown_df = df.copy()
    unknown_df.loc[200, 'C'] = 'U'
    assert card.predict(unknown_df)[200] == pytest.approx(416.4430811052408, FUZZ_THRESHOLD)


def test_card_predict_with_unknown_feature_default_max():
    np.random.seed(9)
    unknown_df = df.copy()
    unknown_df.loc[200, 'C'] = 'U'
    score, sub = card.predict(unknown_df, default='max', return_sub=True)

    assert sub.loc[200, 'C'] == card['C']['scores'].max()
    assert score[200] == pytest.approx(421.3399668050622, FUZZ_THRESHOLD)


def test_card_predict_with_unknown_feature_default_with_value():
    np.random.seed(9)
    unknown_df = df.copy()
    unknown_df.loc[200, 'C'] = 'U'
    score, sub = card.predict(unknown_df, default=42, return_sub=True)

    assert sub.loc[200, 'C'] == 42
    assert score[200] == pytest.approx(327.2767487322905, FUZZ_THRESHOLD)

