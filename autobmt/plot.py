#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: plot.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-10-18
'''
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import six
from matplotlib import gridspec
from openpyxl.drawing.image import Image

from .statistics import calc_var_summary


def ellipsis_fun(x, decimal=2, ellipsis=16):
    if re.search(r'\[(.*?) ~ (.*?)\)', str(x)):
        le = x.split('~')[0]
        ri = x.split('~')[1]
        left = re.match(r'\d+.\[(.*?) ', le)
        right = re.match(r' (.*?)\)', ri)
        if left:
            tmp = round(float(left.groups()[0]), decimal)
            l = f"{le.split('[')[0]}[{tmp} "
            le = l
        if right:
            tmp = round(float(right.groups()[0]), decimal)
            r = f" {tmp})"
            ri = r

        return f"{le} ~ {ri}"
    else:
        return str(x)[:ellipsis] + '..'


def plot_var_bin_summary(frame, cols, target='target', by='type', file_path=None, sheet_name='plot_var_bin_summary',
                         need_bin=False, decimal=2, ellipsis=16, **kwargs):
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
    summary_num = len(summary_df_dict)

    save_jpg_path = None
    if file_path is not None:
        if '.xlsx' in file_path:
            fp_sp = file_path.split('.xlsx')
            if '' == fp_sp[1]:
                save_jpg_path = fp_sp[0] + '_var_jpg'
                os.makedirs(save_jpg_path, exist_ok=True)

            else:
                save_jpg_path = os.path.join(file_path, 'var_jpg')
                os.makedirs(save_jpg_path, exist_ok=True)
                file_path = os.path.join(file_path, 'plot_var_bin_summary.xlsx')
        else:
            save_jpg_path = os.path.join(file_path, 'var_jpg')
            os.makedirs(save_jpg_path, exist_ok=True)
            file_path = os.path.join(file_path, 'plot_var_bin_summary.xlsx')

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
            df['range'] = df['range'].map(lambda x: ellipsis_fun(x, decimal, ellipsis))

            x_point = np.arange(len(df))
            y_point = df['total_pct']

            ax1.bar(x_point, y_point, color='Orange', alpha=0.4, width=0.5, label='PctTotal')
            x_labels = list(df['range'])
            plt.xticks(np.arange(len(df)), x_labels, fontsize=10, rotation=45)

            for x, y in zip(x_point, y_point):
                ax1.text(x + 0.05, y + 0.01, str(round(y * 100, 2)) + '%', ha='center', va='bottom', fontsize=12)
            ax1.set_ylabel('total_pct', fontsize=12)

            ax1.set_ylim([0, max(y_point) + ((max(y_point) - min(y_point)) / len(y_point))])
            bottom, top = ax1.get_ylim()
            ax2 = ax1.twinx()
            ax2.plot(x_point, df['positive_rate'], '-ro', color='red')

            for x, y in zip(x_point, df['positive_rate']):
                ax2.text(x + 0.05, y, str(round(y * 100, 2)) + '%', ha='center', va='bottom', fontsize=12, color='r')
            ax2.set_ylabel('positive_rate', fontsize=12)
            # ax2.set_ylim([0, max(df['positive_rate']) + 0.01])
            ax2.set_ylim(
                [0, max(df['positive_rate']) + ((max(df['positive_rate']) - min(df['positive_rate'])) / len(df))])

            plt.title('{}:{}\nIV: {:.5f}'.format(k, col, df['IV'].iloc[0]), loc='right', fontsize='small')
            # 将数据详情表添加
            tmp_df = df[['range', 'woe', 'iv', 'total']]
            round_cols = ['woe', 'iv']
            tmp_df[round_cols] = tmp_df[round_cols].applymap(lambda v: round(v, 4) if pd.notnull(v) else '')
            mpl_table = plt.table(cellText=tmp_df.values, colLabels=tmp_df.columns,
                                  colWidths=[0.2, 0.1, 0.1, 0.1],
                                  loc='top')  # loc='top'将详情放到顶部

            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(5)

            header_color = 'darkorange'
            row_colors = ['bisque', 'w']
            header_columns = 0

            for k, cell in six.iteritems(mpl_table._cells):
                cell.set_edgecolor('w')
                if k[0] == 0 or k[1] < header_columns:
                    cell.set_text_props(weight='bold', color='w')
                    cell.set_facecolor(header_color)
                else:
                    cell.set_facecolor(row_colors[k[0] % len(row_colors)])

        plt.tight_layout()

        if save_jpg_path is not None:  # 判断是否需要保存
            plt.savefig(os.path.join(save_jpg_path, '{}.png'.format(col)), dpi=300,bbox_inches='tight')  # dpi控制清晰度
        # plt.show()

    # if save_jpg_path is not None:
    #     workbook = xlsxwriter.Workbook(file_path)
    #     worksheet = workbook.add_worksheet(sheet_name)
    #
    #     for i, jpg_name in enumerate(cols):
    #         worksheet.insert_image('A{}'.format(i * 29 + 3), os.path.join(save_jpg_path, '{}.jpg'.format(jpg_name),
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
            img = Image(os.path.join(save_jpg_path, '{}.png'.format(jpg_name)))
            newsize = (summary_num*700, 600)
            img.width, img.height = newsize  # 设置图片的宽和高
            sh.add_image(img, 'A{}'.format(i * 35 + 3))
        wb.save(file_path)
