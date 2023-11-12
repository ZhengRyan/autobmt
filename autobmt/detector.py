#!/usr/bin/python

"""Command line tools for detecting csv data

Team: ESC

Examples:

    python detector.py -i xxx.csv -o report.csv

"""

import pandas as pd


def get_top_values(series, top=5, reverse=False):
    """Get top/bottom n values

    Args:
        series (Series): data series
        top (number): number of top/bottom n values
        reverse (bool): it will return bottom n values if True is given

    Returns:
        Series: Series of top/bottom n values and percentage. ['value:percent', None]
    """
    itype = 'top'
    counts = series.value_counts()
    counts = list(zip(counts.index, counts, counts.divide(series.size)))

    if reverse:
        counts.reverse()
        itype = 'bottom'

    template = "{0[0]}:{0[2]:.2%}"
    indexs = [itype + str(i + 1) for i in range(top)]
    values = [template.format(counts[i]) if i < len(counts) else None for i in range(top)]

    return pd.Series(values, index=indexs)


def get_describe(series, percentiles=[.25, .5, .75]):
    """Get describe of series

    Args:
        series (Series): data series
        percentiles: the percentiles to include in the output

    Returns:
        Series: the describe of data include mean, std, min, max and percentiles
    """
    d = series.describe(percentiles)
    return d.drop('count')


def count_blank(series, blanks=[None]):
    """Count number and percentage of blank values in series

    Args:
        series (Series): data series
        blanks (list): list of blank values

    Returns:
        number: number of blanks
        str: the percentage of blank values
    """
    # n = 0
    # counts = series.value_counts()
    # for blank in blanks:
    #     if blank in counts.keys():
    #         n += counts[blank]

    n = series.isnull().sum()

    return (n, "{0:.2%}".format(n / series.size), n / series.size)


def is_numeric(series):
    """Check if the series's type is numeric

    Args:
        series (Series): data series

    Returns:
        bool
    """
    return series.dtype.kind in 'ifc'


def detect(dataframe, dic_name=None):
    """ Detect data

    Args:
        dataframe (DataFrame): data that will be detected

    Returns:
        DataFrame: report of detecting
    """

    rows = []
    for name, series in dataframe.items():
        numeric_index = ['mean', 'std', 'min', '1%', '10%', '50%', '75%', '90%', '99%', 'max']
        discrete_index = ['top1', 'top2', 'top3', 'top4', 'top5', 'bottom5', 'bottom4', 'bottom3', 'bottom2', 'bottom1']

        details_index = [numeric_index[i] + '_or_' + discrete_index[i] for i in range(len(numeric_index))]
        details = []

        if is_numeric(series):
            desc = get_describe(
                series,
                percentiles=[.01, .1, .5, .75, .9, .99]
            )
            details = desc.tolist()
        else:
            top5 = get_top_values(series)
            bottom5 = get_top_values(series, reverse=True)
            details = top5.tolist() + bottom5[::-1].tolist()

        nblank, pblank, pblank_ = count_blank(series)

        ###add 2020/01/02 RyanZheng
        value_max_percent = get_max_percent(series)
        ###add 2020/01/02 RyanZheng

        row = pd.Series(
            index=['type', 'size', 'missing', 'missing_q', 'unique', 'value_max_percent'] + details_index,
            data=[series.dtype, series.size, pblank, pblank_, series.nunique(), value_max_percent] + details
        )

        row.name = name
        rows.append(row)

    # return pd.DataFrame(rows)

    ### add 2020/01/02 RyanZheng
    eda_df = pd.DataFrame(rows)
    if dic_name is not None and isinstance(dic_name, dict):
        # 增加一列中文名称列
        eda_df.insert(0, 'cn', eda_df.index.map(dic_name))
        # eda_df['cn'] = eda_df.index.map(dic_name)
    eda_df.index.name = 'var_name'
    eda_df = eda_df.reset_index()
    eda_df['type'] = eda_df['type'].astype(str)
    return eda_df
    ### add 2020/01/02 RyanZheng


###add 2020/01/02 RyanZheng
def get_max_percent(series):
    """
    获取变量中同一个值出现次数最多的该值的占比
    Args:
        series:

    Returns:

    """
    return max(series.value_counts(dropna=False) / len(series))
###add 2020/01/02 RyanZheng
