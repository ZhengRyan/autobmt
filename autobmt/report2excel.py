#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: report2excel.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-20
'''

import re
import string
import warnings

import xlsxwriter

from .logger_utils import Logger

warnings.filterwarnings('ignore')

log = Logger(level="info", name=__name__).logger


class Report2Excel:
    """
    统计信息写入到excel
    """

    def __init__(self, file_name, workbook=None, current_row=8, row_spaces=3, column_space=2):
        self.file_name = file_name
        self.workbook = self.create_workbook(workbook)
        self.define_global_format()
        self.row_spaces = row_spaces
        self.column_space = column_space

    def create_workbook(self, workbook):
        """类初始化的时候,必须创建或者返回一个workbook对象"""
        if isinstance(workbook, xlsxwriter.workbook.Workbook):
            return workbook
        else:
            workbook = xlsxwriter.Workbook(self.file_name, {'nan_inf_to_errors': True})
        return workbook

    def close_workbook(self):
        """关闭workbook对象,开始写入到excel文件中"""
        self.workbook.close()

    def df_to_excel(self, evaluate_df, sheet_name):
        # excel相关格式
        worksheet = self.workbook.add_worksheet(sheet_name)

        def write_df_to_excel_by_row(df, worksheet, start_row, start_column, str_formats):
            """按列写入df,因为要单独给每一列一种格式"""
            for col_num, column_name in enumerate(df.columns.values):
                try:
                    worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                           str_formats)
                except Exception as e:
                    t = list(map(str, df[column_name].values.tolist()))
                    worksheet.write_column(start_row, start_column + col_num, t,
                                           str_formats)

        # 写入标题
        for col_num, value in enumerate(evaluate_df.columns.values):
            worksheet.write(0, col_num, value, self.header_format)

        # 写入数据和插入图像
        start_row, start_column = 1, 0
        write_df_to_excel_by_row(evaluate_df, worksheet, start_row, start_column, self.str_formats)
        worksheet.set_column(0, 1, 18)
        # workbook.close()

    def define_global_format(self):
        """定义Excel全局格式"""
        self.merge_format = self.workbook.add_format({  # 大标题的格式
            'bold': True,
            'font_name': '微软雅黑',
            'font_size': 20,
            'border': False,  # 边框线
            'align': 'center',  # 水平居中
            'valign': 'vcenter',  # 垂直居中
            'fg_color': '#ffffff',  # 颜色填充
        })
        self.title_format = self.workbook.add_format({  # 标题的格式
            'bold': True,
            'font_name': '微软雅黑',
            'font_size': 15,
            'border': False,  # 边框线
            'valign': 'top',
            # 'fg_color': '#ddebf7'
        })
        self.sub_title_format = self.workbook.add_format({  # 子标题的格式
            'bold': True,
            'font_name': '微软雅黑',
            'font_size': 12,
            'border': False,  # 边框线
            'valign': 'top',
            # 'fg_color': '#ddebf7'
        })
        self.content_format = self.workbook.add_format({  # 文字的格式
            'bold': False,
            'font_name': '微软雅黑',
            'font_size': 10,
            'border': False,  # 边框线
            'valign': 'top',
        })
        self.content_dict_format = self.workbook.add_format({  # 文字的格式
            'bold': True,
            'font_name': '微软雅黑',
            'font_size': 10,
            'border': False,  # 边框线
            'valign': 'top',
            'font_color': '#275b8e',
        })
        self.ps_format = self.workbook.add_format({  # 注释的格式
            'bold': False,
            'font_name': '微软雅黑',
            'font_size': 10,
            'font_color': 'red',
            'border': False,  # 边框线
            'valign': 'top'
        })
        self.header_format = self.workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'center',
            'font_name': 'Consolas',
            'font_size': 11,
            'border': 1,
            'bottom': 2,
        })
        ### 数字格式
        self.left_formats = self.workbook.add_format({'font_name': 'Consolas', 'align': 'left'})
        self.integer_formats = self.workbook.add_format(
            {'font_name': 'Consolas', 'num_format': '#,##0', 'align': 'left'})  # 整数型
        self.float_formats = self.workbook.add_format(
            {'font_name': 'Consolas', 'num_format': '#,##0.00', 'align': 'left'})  # 浮点型保留2位
        self.percentage_formats = self.workbook.add_format(
            {'font_name': 'Consolas', 'num_format': '0.00%', 'align': 'left'})  # 百分数保留2位
        self.varname_formats = self.workbook.add_format(
            {'color': 'blue', 'underline': True, 'font_name': 'Consolas', 'font_size': 11, 'align': 'left'})
        self.str_formats = self.workbook.add_format(
            {'font_name': 'Consolas', 'font_size': 11, 'align': 'left'})
        self.float_formats4 = self.workbook.add_format(
            {'num_format': '#,##0.0000', 'font_name': 'Consolas', 'font_size': 11, 'align': 'left'})  # 浮点型保留4位
        self.font_formats = self.workbook.add_format({'font_name': 'Consolas', 'font_size': 11, 'align': 'left'})  # 其他
        self.float_formats_new = self.workbook.add_format(
            {'font_name': 'Consolas', 'num_format': '_(* #,##0.00_);_(* (#,##0.00);_(* "-"??_);_(@_)',
             'align': 'left'})  # 百分数保留2位


# Standard Library


# Third Party Stuff


EXCEL_RANGE_PATTERN = re.compile(r'([a-zA-Z]+)([\d]+):([a-zA-Z]+)([\d]+)')

XLSXWRITER_FORMAT_PROPERTIES = (
    'font_name',
    'font_size',
    'font_color',
    'bold',
    'italic',
    'underline',
    'font_strikeout',
    'font_script',
    'num_format',
    'locked',
    'hidden',
    'text_h_align',
    'text_v_align',
    'rotation',
    'text_wrap',
    'text_justlast',
    # 'center_across',
    'indent',
    'shrink',
    'pattern',
    'bg_color',
    'fg_color',
    'bottom',
    'top',
    'left',
    'right',
    'bottom_color',
    'top_color',
    'left_color',
    'right_color',
)


def duplicate_xlsxwriter_format_object(workbook, old_format):
    properties = {}
    if old_format is not None:
        for property_name in XLSXWRITER_FORMAT_PROPERTIES:
            properties[property_name] = getattr(old_format, property_name)

    return workbook.add_format(properties)


def col2num(col):
    num = 0
    for c in col:
        if c in string.ascii_letters:
            num = num * 26 + (ord(c.upper()) - ord('A')) + 1
    return num


def excel_range_string_to_indices(range_string):
    try:
        first_col_name, first_row, last_col_name, last_row = EXCEL_RANGE_PATTERN.findall(
            range_string)[0]
    except IndexError:
        raise ValueError("Invalid range string.")

    first_col_index = col2num(first_col_name) - 1
    first_row_index = int(first_row) - 1
    last_col_index = col2num(last_col_name) - 1
    last_row_index = int(last_row) - 1

    return (
        first_col_index,
        first_row_index,
        last_col_index,
        last_row_index
    )


def apply_border_to_cell(workbook, worksheet, row_index, col_index, format_properties):
    try:
        cell = worksheet.table[row_index][col_index]
        new_format = duplicate_xlsxwriter_format_object(workbook, cell.format)

        # Convert properties in the constructor to method calls.
        for key, value in format_properties.items():
            getattr(new_format, 'set_' + key)(value)

        # Update cell object
        worksheet.table[row_index][col_index] = cell = cell._replace(format=new_format)
    except KeyError:
        format = workbook.add_format(format_properties)
        worksheet.write(row_index, col_index, None, format)


def apply_outer_border_to_range(workbook, worksheet, options=None):
    options = options or {}

    border_style = options.get("border_style", 1)
    range_string = options.get("range_string", None)

    if range_string is not None:
        first_col_index, first_row_index, last_col_index, last_row_index = excel_range_string_to_indices(
            range_string)
    else:
        first_col_index = options.get("first_col_index", None)
        last_col_index = options.get("last_col_index", None)
        first_row_index = options.get("first_row_index", None)
        last_row_index = options.get("last_row_index", None)

        all_are_none = all(map(lambda x: x is None, [
            first_col_index,
            last_col_index,
            first_row_index,
            last_row_index,
        ]))

        if all_are_none:
            raise Exception("You need to specify the range")

    for row_index in range(first_row_index, last_row_index + 1):
        left_border = {
            "left": border_style,
        }
        right_border = {
            "right": border_style,
        }

        apply_border_to_cell(workbook, worksheet, row_index, first_col_index, left_border)
        apply_border_to_cell(workbook, worksheet, row_index, last_col_index, right_border)

    for col_index in range(first_col_index, last_col_index + 1):
        top_border = {
            "top": border_style,
        }

        bottom_border = {
            "bottom": border_style,
        }

        apply_border_to_cell(workbook, worksheet, first_row_index, col_index, top_border)
        apply_border_to_cell(workbook, worksheet, last_row_index, col_index, bottom_border)

    top_left_border = {
        "top": border_style,
        "left": border_style,
    }
    apply_border_to_cell(workbook, worksheet, first_row_index, first_col_index, top_left_border)

    top_right_border = {
        "top": border_style,
        "right": border_style,
    }
    apply_border_to_cell(workbook, worksheet, first_row_index, last_col_index, top_right_border)

    bottom_left_border = {
        "bottom": border_style,
        "left": border_style,
    }
    apply_border_to_cell(workbook, worksheet, last_row_index, first_col_index, bottom_left_border)

    bottom_right_border = {
        "bottom": border_style,
        "right": border_style,
    }
    apply_border_to_cell(workbook, worksheet, last_row_index, last_col_index, bottom_right_border)


def var_summary_to_excelold(var_summary_df, workbook=None, sheet_name=None):
    # excel相关格式
    # workbook = xlsxwriter.Workbook(output, {'nan_inf_to_errors': True})
    # worksheet = workbook.add_worksheet('分箱详情')
    worksheet = workbook.add_worksheet(sheet_name)
    # Add a header format.
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'center',
        'font_name': 'Consolas',
        'font_size': 11,
        'border': 1,
        'bottom': 2,
    })
    varname_formats = workbook.add_format(
        {'color': 'blue', 'underline': True, 'font_name': 'Consolas', 'font_size': 11, })
    percentage_formats = workbook.add_format(
        {'num_format': '0.00%', 'font_name': 'Consolas', 'font_size': 11, })  # 百分数保留2位
    pct_bad_formats = workbook.add_format(
        {'num_format': '0.00%', 'font_name': 'Consolas', 'font_size': 11, 'font_color': '#980101'})  # 百分数保留2位
    pct_good_formats = workbook.add_format(
        {'num_format': '0.00%', 'font_name': 'Consolas', 'font_size': 11, 'font_color': '#0050aa'})  # 百分数保留2位
    float_formats4 = workbook.add_format(
        {'num_format': '#,##0.0000', 'font_name': 'Consolas', 'font_size': 11, })  # 浮点型保留4位
    integer_formats = workbook.add_format(
        {'num_format': '#,##0', 'font_name': 'Consolas', 'font_size': 11, })  # 整数型
    font_formats = workbook.add_format({'font_name': 'Consolas', 'font_size': 11, })  # 整数型
    row_formats = workbook.add_format({'bottom': 1})  # 下边框线

    def colnum_string(n):
        """将数字n转成excel的列符号"""
        string = ""
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            string = chr(65 + remainder) + string
        return string

    def write_df_to_excel_by_row(index, feature, df, workbook, worksheet, start_row, start_column):
        """按列写入df,因为要单独给每一列一种格式"""
        for col_num, column_name in enumerate(df.columns.values):
            if column_name in ['positive_rate', 'Pct_Bin']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       percentage_formats)
            elif column_name in ['Pct_Bad']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       pct_bad_formats)
            elif column_name in ['Pct_Good']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       pct_good_formats)
            elif column_name in ['var_name']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       varname_formats)
            elif column_name in ['iv', 'IV']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       float_formats4)
            elif column_name in ['Total', 'Bad', 'Good']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       integer_formats)
            elif column_name in ['woe']:
                # 输入到Excel中的时候，将WoE的值放大100倍，方便观察
                # worksheet.write_column(start_row, start_column + col_num,
                #                        # (df[column_name] * 100).astype(int).values.tolist(),
                #                        integer_formats)
                worksheet.write_column(start_row, start_column + col_num,
                                       df[column_name].values.tolist(),
                                       float_formats4)
            else:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       font_formats)

        # 给最后一行加上下框线,用于区分变量
        end_row, end_column = start_row + df.shape[0], df.shape[1]
        worksheet.conditional_format('A{end_row}:{end_column}{end_row}' \
                                     .format(end_row=end_row, end_column=colnum_string(end_column)), \
                                     {'type': 'no_blanks', 'format': row_formats})

        # 给BadRate列加上条件格式
        badrate_index = colnum_string(start_column + df.columns.get_loc('Bad_Rate') + 1)
        worksheet.conditional_format(
            '{c}{start_row}:{c}{end_row}'.format(start_row=start_row, end_row=end_row, c=badrate_index),
            {'type': 'data_bar', 'bar_color': '#f0c4c4', 'bar_solid': True})

        # 给每个分箱变量,绘制badrate曲线和woe分布图
        # badrate
        graph = workbook.add_chart({'type': 'line'})
        range_column_index = df.columns.get_loc('range') + start_column
        badrate_column_index = df.columns.get_loc('Bad_Rate') + start_column

        range_categories = [worksheet.get_name(), start_row, range_column_index, end_row - 1,
                            range_column_index]
        range_value = [worksheet.get_name(), start_row, badrate_column_index, end_row - 1,
                       badrate_column_index]
        graph.add_series({'name': 'BadRate', 'categories': range_categories, 'values': range_value,
                          'marker': {'type': 'circle'}, 'data_labels': {'value': True}})
        graph.set_size({'width': 450, 'height': 200})
        graph.set_title(
            {'name': 'Bad Rate - {}'.format(feature), 'overlay': False,
             'name_font': {'name': '微软雅黑', 'size': 9, 'bold': True}})
        graph.set_x_axis({'line': {'none': True}})
        graph.set_y_axis({'line': {'none': True}})
        graph.set_legend({'none': True})  # 设置图例
        worksheet.insert_chart(start_row + index * 7, end_column + 1, graph)

        # WoE
        column_graph = workbook.add_chart({'type': 'column'})
        range_column_index = df.columns.get_loc('range') + start_column
        badrate_column_index = df.columns.get_loc('woe') + start_column

        range_categories = [worksheet.get_name(), start_row, range_column_index, end_row - 1,
                            range_column_index]
        range_value = [worksheet.get_name(), start_row, badrate_column_index, end_row - 1,
                       badrate_column_index]
        column_graph.add_series({'name': 'woe', 'categories': range_categories, 'values': range_value,
                                 'marker': {'type': 'circle'}, 'data_labels': {'value': True}})
        column_graph.set_size({'width': 450, 'height': 200})  # TODO 大小改成和单元格的大小一致,以单元格的大小为单位
        column_graph.set_title(
            {'name': 'WoE - {}'.format(feature), 'overlay': False,
             'name_font': {'name': '微软雅黑', 'size': 9, 'bold': True}})
        column_graph.set_x_axis({'line': {'none': True}, 'label_position': 'low'})
        column_graph.set_y_axis({'line': {'none': True}})
        column_graph.set_legend({'none': True})  # 设置图例
        worksheet.insert_chart(start_row + index * 7, end_column + 9, column_graph)

        # 给变量名称插入跳转超链接
        for i in range(1, df.shape[0] + 1):
            worksheet.write_url('B{}'.format(start_row + i),
                                'internal:{}!M{}:AC{}'.format(worksheet.get_name(), start_row + index * 7 + 1,
                                                              start_row + index * 7 + 10), string=feature,
                                tip='Jump to charts')

    # 写入标题
    for col_num, value in enumerate(var_summary_df.columns.values):
        worksheet.write(0, col_num, value, header_format)

    # 写入数据和插入图像
    start_row, start_column = 1, 0
    index = 0
    for name, single_df in var_summary_df.groupby('var_name'):
        write_df_to_excel_by_row(index, name, single_df, workbook, worksheet, start_row, start_column)
        start_row = single_df.shape[0] + start_row
        index += 1

    # 冻结窗格
    worksheet.freeze_panes(1, 2)
    worksheet.set_column(0, 1, 18)

    # # 保存为excel
    # workbook.close()


def var_summary_to_excel(var_summary_df, workbook=None, sheet_name=None):
    # excel相关格式
    # workbook = xlsxwriter.Workbook(output, {'nan_inf_to_errors': True})
    # worksheet = workbook.add_worksheet('分箱详情')
    worksheet = workbook.add_worksheet(sheet_name)
    # Add a header format.
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'center',
        'font_name': 'Consolas',
        'font_size': 11,
        'border': 1,
        'bottom': 2,
    })
    varname_formats = workbook.add_format(
        {'color': 'blue', 'underline': True, 'font_name': 'Consolas', 'font_size': 11, })
    percentage_formats = workbook.add_format(
        {'num_format': '0.00%', 'font_name': 'Consolas', 'font_size': 11, })  # 百分数保留2位
    positive_pct_formats = workbook.add_format(
        {'num_format': '0.00%', 'font_name': 'Consolas', 'font_size': 11, 'font_color': '#980101'})  # 百分数保留2位
    negative_pct_formats = workbook.add_format(
        {'num_format': '0.00%', 'font_name': 'Consolas', 'font_size': 11, 'font_color': '#0050aa'})  # 百分数保留2位
    float_formats4 = workbook.add_format(
        {'num_format': '#,##0.0000', 'font_name': 'Consolas', 'font_size': 11, })  # 浮点型保留4位
    integer_formats = workbook.add_format(
        {'num_format': '#,##0', 'font_name': 'Consolas', 'font_size': 11, })  # 整数型
    font_formats = workbook.add_format({'font_name': 'Consolas', 'font_size': 11, })  # 整数型
    row_formats = workbook.add_format({'bottom': 1})  # 下边框线

    def colnum_string(n):
        """将数字n转成excel的列符号"""
        string = ""
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            string = chr(65 + remainder) + string
        return string

    def write_df_to_excel_by_row(index, feature, df, workbook, worksheet, start_row, start_column):
        """按列写入df,因为要单独给每一列一种格式"""
        for col_num, column_name in enumerate(df.columns.values):
            if column_name in ['positive_rate', 'total_pct']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       percentage_formats)
            elif column_name in ['positive_pct']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       positive_pct_formats)
            elif column_name in ['negative_pct']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       negative_pct_formats)
            elif column_name in ['var_name']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       varname_formats)
            elif column_name in ['iv', 'IV']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       float_formats4)
            elif column_name in ['total', 'positive', 'negative']:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       integer_formats)
            elif column_name in ['woe']:
                # 输入到Excel中的时候，将WoE的值放大100倍，方便观察
                # worksheet.write_column(start_row, start_column + col_num,
                #                        # (df[column_name] * 100).astype(int).values.tolist(),
                #                        integer_formats)
                worksheet.write_column(start_row, start_column + col_num,
                                       df[column_name].values.tolist(),
                                       float_formats4)
            else:
                worksheet.write_column(start_row, start_column + col_num, df[column_name].values.tolist(),
                                       font_formats)

        # 给最后一行加上下框线,用于区分变量
        end_row, end_column = start_row + df.shape[0], df.shape[1]
        worksheet.conditional_format('A{end_row}:{end_column}{end_row}' \
                                     .format(end_row=end_row, end_column=colnum_string(end_column)), \
                                     {'type': 'no_blanks', 'format': row_formats})

        # 给BadRate列加上条件格式
        badrate_index = colnum_string(start_column + df.columns.get_loc('positive_rate') + 1)
        worksheet.conditional_format(
            '{c}{start_row}:{c}{end_row}'.format(start_row=start_row, end_row=end_row, c=badrate_index),
            {'type': 'data_bar', 'bar_color': '#f0c4c4', 'bar_solid': True})

        # 给每个分箱变量,绘制badrate曲线和woe分布图
        # badrate
        graph = workbook.add_chart({'type': 'line'})
        range_column_index = df.columns.get_loc('range') + start_column
        badrate_column_index = df.columns.get_loc('positive_rate') + start_column

        range_categories = [worksheet.get_name(), start_row, range_column_index, end_row - 1,
                            range_column_index]
        range_value = [worksheet.get_name(), start_row, badrate_column_index, end_row - 1,
                       badrate_column_index]
        graph.add_series({'name': 'positive_rate', 'categories': range_categories, 'values': range_value,
                          'marker': {'type': 'circle'}, 'data_labels': {'value': True}})
        graph.set_size({'width': 450, 'height': 200})
        graph.set_title(
            {'name': 'positive_rate - {}'.format(feature), 'overlay': False,
             'name_font': {'name': '微软雅黑', 'size': 9, 'bold': True}})
        graph.set_x_axis({'line': {'none': True}})
        graph.set_y_axis({'line': {'none': True}})
        graph.set_legend({'none': True})  # 设置图例
        worksheet.insert_chart(start_row + index * 7, end_column + 1, graph)

        # WoE
        column_graph = workbook.add_chart({'type': 'column'})
        range_column_index = df.columns.get_loc('range') + start_column
        badrate_column_index = df.columns.get_loc('woe') + start_column

        range_categories = [worksheet.get_name(), start_row, range_column_index, end_row - 1,
                            range_column_index]
        range_value = [worksheet.get_name(), start_row, badrate_column_index, end_row - 1,
                       badrate_column_index]
        column_graph.add_series({'name': 'woe', 'categories': range_categories, 'values': range_value,
                                 'marker': {'type': 'circle'}, 'data_labels': {'value': True}})
        column_graph.set_size({'width': 450, 'height': 200})  # TODO 大小改成和单元格的大小一致,以单元格的大小为单位
        column_graph.set_title(
            {'name': 'WoE - {}'.format(feature), 'overlay': False,
             'name_font': {'name': '微软雅黑', 'size': 9, 'bold': True}})
        column_graph.set_x_axis({'line': {'none': True}, 'label_position': 'low'})
        column_graph.set_y_axis({'line': {'none': True}})
        column_graph.set_legend({'none': True})  # 设置图例
        worksheet.insert_chart(start_row + index * 7, end_column + 9, column_graph)

        # 给变量名称插入跳转超链接
        for i in range(1, df.shape[0] + 1):
            worksheet.write_url('B{}'.format(start_row + i),
                                'internal:{}!M{}:AC{}'.format(worksheet.get_name(), start_row + index * 7 + 1,
                                                              start_row + index * 7 + 10), string=feature,
                                tip='Jump to charts')

    # 写入标题
    for col_num, value in enumerate(var_summary_df.columns.values):
        worksheet.write(0, col_num, value, header_format)

    # 写入数据和插入图像
    start_row, start_column = 1, 0
    index = 0
    for name, single_df in var_summary_df.groupby('var_name'):
        write_df_to_excel_by_row(index, name, single_df, workbook, worksheet, start_row, start_column)
        start_row = single_df.shape[0] + start_row
        index += 1

    # 冻结窗格
    worksheet.freeze_panes(1, 2)
    worksheet.set_column(0, 1, 18)

    # # 保存为excel
    # workbook.close()


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import os

    # 创建一个数据
    df = pd.DataFrame(np.random.randn(182, 9), columns=list('ABCDEFGHI'))
    column_list = df.columns
    # 使用XlsxWriter引擎创建一个pandas Excel writer。
    writer = pd.ExcelWriter(os.path.join('..','tests','test_report2excel.xlsx'), engine='xlsxwriter')

    df.to_excel(writer, index=False)

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    ###########################################
    # worksheet.set_landscape()
    # worksheet.set_paper(8)
    # worksheet.set_margins(0.787402, 0.787402, 0.5, 0.787402)

    apply_outer_border_to_range(
        workbook,
        worksheet,
        {
            "range_string": "C10:M20",
            "border_style": 5,
        },
    )

    # 关闭workbook
    workbook.close()
