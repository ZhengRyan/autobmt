#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: scorecard.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-20
'''

import re
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from .transformer import FeatureBin, WoeTransformer
from .utils import to_ndarray, save_json, load_json

RE_NUM = r'-?\d+(.\d+)?'
RE_SEP = r'[~-]'
RE_BEGIN = r'(-inf|{num})'.format(num=RE_NUM)
RE_END = r'(inf|{num})'.format(num=RE_NUM)
RE_RANGE = r'\[{begin}\s*{sep}\s*{end}\)'.format(
    begin=RE_BEGIN,
    end=RE_END,
    sep=RE_SEP,
)

NUMBER_EXP = re.compile(RE_RANGE)

NUMBER_EMPTY = -9999999
NUMBER_INF = 1e10
FACTOR_EMPTY = 'MISSING'
FACTOR_UNKNOWN = 'UNKNOWN'
ELSE_GROUP = 'else'


class ScoreCard(BaseEstimator):
    def __init__(self, pdo=50, rate=2, odds=15, base_score=600,
                 card={}, combiner={}, transer=None, AB={}, **kwargs):
        """

        Args:
            pdo (int):point double odds;;;当odds增加一倍，评分增加的分数
            rate (int):
            odds (int): odds at base point;;;基准分值对应的odds
            base_score (int): base point;;;基准分数
            card (dict): 评分卡
            combiner (autobmt.FeatureBin): 分箱规则
            transer (autobmt.WoeTransformer): 变量分箱对应的woe值
            **kwargs:
        """
        ##实际意义为当比率为1/15，输出基准评分600，当比率为基准比率2倍时，1/7.5，基准分下降50分，为550
        self.pdo = pdo  # point double odds;;;当odds增加一倍，评分增加的分数
        self.rate = rate  #
        self.odds = odds  # odds at base point;;;基准分值对应的odds
        self.base_score = base_score  # base point;;;基准分数
        self.AB = AB    #自定义的大A，大B

        if AB:
            self.factor = self.AB['B']
            self.offset = self.AB['A']
        else:
            self.factor = pdo / np.log(rate)  # 大B;;;B=72.13475204444818
            self.offset = base_score - (pdo / np.log(rate)) * np.log(odds)  # 大A;;;A=404.65547021957406

        self._combiner = combiner
        self.transer = transer
        self.model = LogisticRegression(**kwargs)

        self._feature_names = None

        self.card = card
        if card:
            self.load(card)

    def __len__(self):
        return len(self.card.keys())

    def __contains__(self, key):
        return key in self.card

    def __getitem__(self, key):
        return self.card[key]

    def __setitem__(self, key, value):
        self.card[key] = value

    def __iter__(self):
        return iter(self.card)

    @property
    def coef_(self):
        """ 逻辑回归模型系数
        """
        return self.model.coef_[0]

    @property
    def intercept_(self):
        """ 逻辑回归模型截距
        """
        return self.model.intercept_[0]

    @property
    def n_features_(self):
        """ 变量个数
        """
        return (self.coef_ != 0).sum()

    @property
    def features_(self):
        """ 变量列表
        """
        if not self._feature_names:
            self._feature_names = list(self.card.keys())

        return self._feature_names

    @property
    def combiner(self):
        if not self._combiner:
            # 如果不存在，则生成新的分箱器
            rules = {}
            for key in self.card:
                rules[key] = self.card[key]['bins']

                self._combiner = FeatureBin().manual_bin(rules)

        return self._combiner

    def fit(self, X, y):
        """
        Args:
            X (2D DataFrame): 变量
            Y (array-like): 目标变量列表
        """
        self._feature_names = X.columns.tolist()

        for f in self.features_:
            if f not in self.transer:
                raise Exception('column \'{f}\' is not in transer'.format(f=f))

        self.model.fit(X, y)
        self.card = self._generate_rules()

        # keep sub_score-median of each feature, as `base_effect` for reason-calculation
        sub_score = self.woe_to_score(X)
        # self.base_effect = pd.Series(
        #     np.median(sub_score, axis=0),
        #     index=self.features_
        # )

        return self

    def predict(self, X, **kwargs):
        """预测分数
        Args:
            X (2D-DataFrame|dict): 需要去预测的变量
            return_sub (Bool): 是否需要返回特征中每个箱子的得分
            default (str|number): 未知特征的默认分数，' min '(默认)，' max '

        Returns:
            array-like: 预测的分数
            DataFrame|dict: 每个特征对应的分数
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.features_]

        bins = self.combiner.transform(X)
        res = self.bin_to_score(bins, **kwargs)
        return res

    def bin_to_score(self, bins, return_sub=False, default='min'):
        """
        通过分箱值直接算分
        Args:
            bins (2D-DataFrame|dict): 使用分箱值替换后的变量
            return_sub (bool): 是否需要返回特征中每个箱子的得分
            default (bool): 未知特征的默认分数，' min '(默认)，' max '

        Returns:

        """
        score = 0
        res = bins.copy()
        for col, rule in self.card.items():
            s_map = rule['scores']
            b = bins[col]

            # set default value for empty group
            default_value = default
            if default == 'min':
                default_value = np.min(s_map)
            elif default == 'max':
                default_value = np.max(s_map)
            elif isinstance(default, str):
                raise ValueError(f'default `{default}` is not valid, only support `min`, `max` or number')

            # append default value to the end of score map
            s_map = np.append(s_map, default_value)

            # # set default group to min score
            # if np.isscalar(b):
            #     b = np.argmin(s_map) if b == self.EMPTY_BIN else b
            # else:
            #     b[b == self.EMPTY_BIN] = np.argmin(s_map)

            # replace score
            res[col] = s_map[b]
            score += s_map[b]

        if return_sub:
            return score, res
        else:
            return score

    def predict_proba(self, X):
        """
        预测概率
        Args:
            X (2D array-like): 需要去预测的变量

        Returns:
            2d array: 预测的概率值（包括正样本和负样本的概率值）
        """
        proba = self.score_to_proba(self.predict(X))
        return np.stack((1 - proba, proba), axis=1)

    def _generate_rules(self):
        if not self._check_rules(self.combiner, self.transer):
            raise Exception('generate failed')

        rules = {}

        for idx, key in enumerate(self.features_):
            weight = self.coef_[idx]

            if weight == 0:
                continue

            # woe = self.transer[key]['woe']
            woe = list(self.transer[key].values())

            rules[key] = {
                'bins': self.combiner[key],
                'woes': woe,
                'weight': weight,
                'scores': self.woe_to_score(woe, weight=weight),
            }

        return rules

    def _check_rules(self, combiner, transer):
        for col in self.features_:
            if col not in combiner:
                raise Exception('column \'{col}\' is not in combiner'.format(col=col))

            if col not in transer:
                raise Exception('column \'{col}\' is not in transer'.format(col=col))

            l_c = len(combiner.export(bin_format=False)[col])
            # l_t = len(transer[col]['woe'])
            l_t = len(transer[col].values())

            if l_c == 0:
                continue

            if np.issubdtype(combiner[col].dtype, np.number):
                if l_c != l_t - 1:
                    raise Exception(
                        'column \'{col}\' is not matched, assert {l_t} bins but given {l_c}'.format(col=col, l_t=l_t,
                                                                                                    l_c=l_c + 1))
            else:
                if l_c != l_t:
                    raise Exception(
                        'column \'{col}\' is not matched, assert {l_t} bins but given {l_c}'.format(col=col, l_t=l_t,
                                                                                                    l_c=l_c))

        return True

    def proba_to_score(self, prob):
        """概率转分

        odds = (1 - prob) / prob    #good:bad
        score = factor * log(odds) + offset

        odds = prob / (1 - prob)    #bad:good
        score = offset - factor * log(odds)

        log(odds) = intercept+woe1*coef1+woe2*coef2+woe3*coef3
        """
        # return self.factor * np.log((1 - prob) / prob) + self.offset
        return self.offset - self.factor * np.log(prob / (1 - prob))

    def woe_sum_to_score(self, woe, weight=None):
        """通过woe计算分

        odds = (1 - prob) / prob    #good:bad
        score = factor * log(odds) + offset

        odds = prob / (1 - prob)    #bad:good
        score = offset - factor * log(odds)

        log(odds) = intercept+woe1*coef1+woe2*coef2+woe3*coef3
        """
        woe = to_ndarray(woe)

        if weight is None:
            weight = self.coef_

        z_cols = weight * woe
        z = self.intercept_ + np.sum(z_cols, axis=1)

        return self.offset - self.factor * z

    def score_to_proba(self, score):
        """分转概率

        Returns:
            array-like|float: 正样本的概率【即：1的概率】
        """
        return 1 / (1 + np.exp((score - self.offset) / self.factor))

    def woe_to_score(self, woe, weight=None):
        """通过woe计算分
        score = A - Blog(odds) = A - B( β0 + β1x1 + … βnxn) = (A - Bβ0) - Bβ1 x1 - … Bβn xn = sum((A - Bβ0)/n - Bβ1 x1 - … Bβn xn)
        """
        woe = to_ndarray(woe)

        if weight is None:
            weight = self.coef_

        b = (self.offset - self.factor * self.intercept_) / self.n_features_  # (A - Bβ0)/n
        s = -self.factor * weight * woe  # -B*βn*xn

        # drop score whose weight is 0
        mask = 1
        if isinstance(weight, np.ndarray):
            mask = (weight != 0).astype(int)

        return (s + b) * mask

    def export(self, to_dataframe=False, to_json=None, to_csv=None, decimal=2):
        """生成一个评分卡对象

        Args:
            to_dataframe (bool): 生成的评分卡是1个DataFrame
            to_json (str|IOBase): 生成的评分卡写出json文件
            to_csv (filepath|IOBase): 生成的评分卡写出csv文件

        Returns:
            dict: 评分卡
        """
        card = dict()
        combiner = self.combiner.export(bin_format=True, index=False)

        for col, rule in self.card.items():
            s_map = rule['scores']
            bins = combiner[col]
            woe_map = np.zeros(len(bins))
            if 'woes' in rule:
                woe_map = rule['woes']
            weight = np.nan
            if 'weight' in rule:
                weight = rule['weight']
            card[col] = dict()

            for i in range(len(bins)):
                # card[col][bins[i]] = round(s_map[i], decimal)
                card[col][bins[i]] = [round(s_map[i], decimal), woe_map[i], weight]

        if to_json is not None:
            save_json(card, to_json)

        if to_dataframe or to_csv is not None:
            rows = list()
            for feature in card:
                for value, score in card[feature].items():
                    rows.append({
                        'feature': feature,
                        'value': value,
                        'score': score[0],
                        'woe': score[1],
                        'weight': score[2],
                    })

            card = pd.DataFrame(rows)

        if to_csv is not None:
            return card.to_csv(to_csv)

        return card

    def _is_numeric(self, bins):
        m = NUMBER_EXP.match(bins[0])

        return m is not None

    def _numeric_parser(self, bins):
        l = list()

        for item in bins:

            # if re.compile('{}.nan'.format(RE_NUM)).match(item):
            if item == 'nan':
                l.append(np.nan)
                continue

            m = NUMBER_EXP.match(item)
            split = m.group(3)

            if split == 'inf':
                # split = np.inf
                continue

            split = float(split)

            l.append(split)

        return np.array(l)

    def parse_bins(self, bins):
        """解析格式化的分箱值
        """
        if self._is_numeric(bins):
            return self._numeric_parser(bins)

        l = list()

        for item in bins:
            if item == ELSE_GROUP:
                l.append(item)
            else:
                l.append(item.split(','))

        return np.array(l, dtype=object)

    def _parse_rule(self, rule):
        bins = self.parse_bins(list(rule.keys()))
        v = list(rule.values())
        if isinstance(v[0], list):
            scores = np.array([i[0] for i in v])
        else:
            scores = np.array(v)

        return {
            'bins': bins,
            'scores': scores,
        }

    def load(self, card=None):
        """
        加载评分卡
        Args:
            card: 从dict或json文件加载评分卡

        Returns:

        """
        card = deepcopy(card)
        if not isinstance(card, dict):
            card = load_json(card)

        for feature in card:
            card[feature] = self._parse_rule(card[feature])

        self.card = card
        self._combiner = {}

        return self
