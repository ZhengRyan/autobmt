#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: __init__.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2022-07-09
'''

from .feature_binning import bin_method_run, chi_bin, dt_bin, equal_freq_bin, kmeans_bin, best_binning
from .detector import detect
from .feature_selection import FeatureSelection, feature_select, stepwise_del_feature
from .report2excel import Report2Excel, var_summary_to_excel
from .statistics import calc_bin_summary, calc_var_summary, compare_inflection_point, get_vif
from .stepwise import stepwise, StatsModel
from .transformer import WoeTransformer, FeatureBin
from .metrics import psi, get_auc_ks_psi, get_auc, get_ks
from .utils import del_df, dump_to_pkl, load_from_pkl, fea_woe_dict_format, to_score
from .logger_utils import Logger
from .scorecard import ScoreCard
from .bayes_opt_tuner import classifiers_model_auto_tune_params
from .plot import plot_var_bin_summary

__version__ = "0.1.8"
VERSION = __version__
