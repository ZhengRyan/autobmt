#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: plot.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-10-18
'''
import os

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import six
from matplotlib import gridspec
from openpyxl.drawing.image import Image

from .statistics import calc_var_summary


def plot_var_bin_summary(frame, cols, target='target', by='type', file_path=None, sheet_name='plot_var_bin_summary',
                         need_bin=False, **kwargs):
    """
    画变量分箱图
    Args:
        frame (DataFrame):
        cols (str|list): 需要画图的变量
        target (str): 目标变量
        by (str): 分开统计的列名
        file_path (str): 保存路径
        sheet_name (str):保存excel的sheet名称
        need_bin (bool):是否要进行分箱

    Returns:

    """
    # frame = {'train':tr_df,'test':te_df,'oot':oot_df}

    if isinstance(cols, str):
        cols = [cols]

    if isinstance(frame, pd.DataFrame):
        summary_df_dict = {}
        if by in frame:
            for name, df in frame.groupby(by):
                summary_df_dict[name] = calc_var_summary(df, include_cols=cols, target=target, need_bin=need_bin,
                                                         **kwargs)
        else:
            summary_df_dict['all'] = calc_var_summary(frame, include_cols=cols, target=target, need_bin=need_bin,
                                                      **kwargs)
    else:
        summary_df_dict = frame

    save_jpg_path = None
    if file_path is not None:
        if '.xlsx' in file_path:
            fp_sp = file_path.split('.xlsx')
            if '' == fp_sp[1]:
                save_jpg_path = fp_sp[0] + '_var_jpg'
                os.makedirs(save_jpg_path, exist_ok=True)

            else:
                save_jpg_path = file_path + '/var_jpg'
                os.makedirs(save_jpg_path, exist_ok=True)
                file_path = file_path + '/plot_var_bin_summary.xlsx'
        else:
            save_jpg_path = file_path + '/var_jpg'
            os.makedirs(save_jpg_path, exist_ok=True)
            file_path = file_path + '/plot_var_bin_summary.xlsx'

    for col in cols:
        # 做图
        # plt.figure(figsize=(15, 4))
        # gs = gridspec.GridSpec(1, 2)
        # gs = gridspec.GridSpec(1, 3)

        gs_num = len(summary_df_dict)

        if gs_num == 1:
            plt.figure(figsize=(gs_num * 5, 4), dpi=400)
        else:
            plt.figure(figsize=(gs_num * 5, 4))

        gs = gridspec.GridSpec(1, gs_num)

        for i, k in enumerate(summary_df_dict):

            df = summary_df_dict[k]

            if i == 0:
                ax1 = plt.subplot(gs[0, 0])
            else:
                ax1 = plt.subplot(gs[0, i], sharey=ax1)

            df = df[df['var_name'] == col]

            x_PctTotal = np.arange(0, len(df), 1)
            y_PctTotal = df['total_pct']

            ax1.bar(x_PctTotal, y_PctTotal, color='lightskyblue', alpha=0.4, width=0.5, label='PctTotal')

            for x, y in zip(x_PctTotal, y_PctTotal):
                ax1.text(x + 0.05, y + 0.01, str(round(y * 100, 2)) + '%', ha='center', va='bottom', fontsize=12)
            ax1.set_ylabel('total_pct', fontsize=12)

            ax1.set_ylim([0, max(y_PctTotal) + 0.3])
            ax2 = ax1.twinx()
            ax2.plot(x_PctTotal, df['positive_rate'], '-ro', color='red')

            for x, y in zip(x_PctTotal, df['positive_rate']):
                ax2.text(x + 0.05, y, str(round(y * 100, 2)) + '%', ha='center', va='bottom', fontsize=12, color='r')
            ax2.set_ylabel('positive_rate', fontsize=12)
            ax2.set_ylim([0, max(df['positive_rate']) + 0.01])

            # title_map = {0: 'train', 1: 'test', 2: 'oot'}
            # plt.title('{}:{}'.format(title_map[i], col))
            plt.title('{}:{}\nIV: {:.5f}'.format(k, col, df['IV'].iloc[0]))
            # plt.title('train:' + col)
            # 将数据详情表添加
            tmp_df = df[['range', 'woe', 'iv', 'total', 'positive_rate']]
            round_cols = ['woe', 'iv', 'positive_rate']
            tmp_df[round_cols] = tmp_df[round_cols].applymap(lambda v: round(v, 4) if pd.notnull(v) else '')
            mpl_table = plt.table(cellText=tmp_df.values, colLabels=tmp_df.columns,
                                  colWidths=[0.15, 0.1, 0.1, 0.1, 0.15],
                                  loc=1)  # loc='top'将详情放到顶部

            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(5)

            header_color = '#40466e'
            row_colors = ['#f1f1f2', 'w']
            header_columns = 0

            for k, cell in six.iteritems(mpl_table._cells):
                cell.set_edgecolor('w')
                if k[0] == 0 or k[1] < header_columns:
                    cell.set_text_props(weight='bold', color='w')
                    cell.set_facecolor(header_color)
                else:
                    cell.set_facecolor(row_colors[k[0] % len(row_colors)])

            # ax1.text(*(ax1.get_xlim()[0], ax1.get_ylim()[1]), 'IV: {:.5f}'.format(df['iv'].sum()), fontsize='x-large')

        plt.tight_layout()

        if save_jpg_path is not None:  # 判断是否需要保存
            plt.savefig(save_jpg_path + '/{}.jpg'.format(col), dpi=300)  # dpi控制清晰度
        # plt.show()

    # if save_jpg_path is not None:
    #     workbook = xlsxwriter.Workbook(file_path)
    #     worksheet = workbook.add_worksheet(sheet_name)
    #
    #     for i, jpg_name in enumerate(cols):
    #         worksheet.insert_image('A{}'.format(i * 29 + 3), save_jpg_path + '/{}.jpg'.format(jpg_name),
    #                                {'x_scale': 1.5, 'y_scale': 1.5})
    #
    #     workbook.close()
    if save_jpg_path is not None:
        try:
            wb = openpyxl.load_workbook(file_path)
        except:
            wb = openpyxl.Workbook()
        sh = wb.create_sheet(sheet_name)
        for i, jpg_name in enumerate(cols):
            img = Image(save_jpg_path + '/{}.jpg'.format(jpg_name))
            newsize = (1900, 500)
            img.width, img.height = newsize  # 设置图片的宽和高
            sh.add_image(img, 'A{}'.format(i * 29 + 3))
        wb.save(file_path)
