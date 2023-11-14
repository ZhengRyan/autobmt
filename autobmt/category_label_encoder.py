#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: category_label_encoder.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-09-21
'''

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
#
#
# def category_to_labelencoder(data, labelencoder=[]):
#     label_encoder_dict = {}
#     le = LabelEncoder()
#     for col in labelencoder:
#         print('{} in process!!!'.format(col))
#         data[col] = le.fit_transform(data[col].values)
#         number = [i for i in range(0, len(le.classes_))]
#         key = list(le.inverse_transform(number))
#         label_encoder_dict[col] = dict(zip(key, number))
#     return label_encoder_dict
#
#
# def category_to_labelencoder_apply(data, labelencoder_dict={}):
#     for col, mapping in labelencoder_dict.items():
#         print('{} in process!!!'.format(col))
#         data[col] = data[col].map(mapping).fillna(-1)
#         data[col] = data[col].astype(int)
#
#
# #####训练
# fruit_data = pd.DataFrame({
#     'fruit': ['apple', 'orange', 'pear', 'orange', 'red'],
#     'color': ['red', 'orange', 'green', 'green', 'red'],
#     'weight': [5, 6, 3, 4, 2]
# })
# print(fruit_data)
#
# labelencoder_cols = ['fruit', 'color']
#
# label_encoder_dict = category_to_labelencoder(fruit_data, labelencoder_cols)
# print(fruit_data)
#
# #####应用
# test_data = pd.DataFrame({
#     'fruit': ['apple', 'orange', 'pear', 'orange', 'red'],
#     'color': ['aaa', 'orange', 'green', 'green', 'red'],
#     'weight': [5, 6, 3, 4, 2]
# })
# print(test_data)
#
# category_to_labelencoder_apply(test_data, label_encoder_dict)
# print(test_data)


print('########################'*5)

import pandas as pd

df = pd.DataFrame([
    ['green', 'Chevrolet', 2017],
    ['blue', 'BMW', 2015],
    ['yellow', 'Lexus', 2018],
])
df.columns = ['color', 'make', 'year']
df.to_csv('df_source.csv')

df_processed = pd.get_dummies(df, prefix_sep="_", columns=df.columns[:-1])
print(df_processed.head(10))
df_processed.to_csv('df_processed.csv')

aa = ['aa', 'bb', 'cc']
print(aa)
aa.remove('aa')
print(aa)

df = pd.DataFrame([
    ['green;yellow;aaa', 'Chevrolet', 2017],
    ['blue;green', 'BMW', 2015],
    ['yellow', 'Lexus', 2018],
])
df.columns = ['color', 'make', 'year']
print(df)

print(df.set_index(["make", "year"])["color"].str.split(";", expand=True))
df = df.set_index(["make", "year"])["color"].str.split(";", expand=True).reset_index()
print(df)
del df['make']
print(pd.get_dummies(df,prefix='', prefix_sep='').groupby(level=0, axis=1).max())


df = pd.DataFrame([
    ['green;yellow;aaa', 'Chevrolet;Lexus', 2017],
    ['blue;green', 'BMW', 2015],
    ['yellow', 'Lexus;BMW', 2018],
])
df.columns = ['color', 'make', 'year']
print(df)

print(df.set_index(["year"])["color"].str.split(";", expand=True))
df = df.set_index(["year"])["color"].str.split(";", expand=True).reset_index()
print(df)
print(pd.get_dummies(df,prefix='', prefix_sep='').groupby(level=0, axis=1).max())
exit(0)
# df = df.set_index(["make", "year"])["color"].str.split(";", expand=True).stack().reset_index(drop=True, level=-1).reset_index().rename(columns={0: "color"})
# print(df)
# del df['make']
# print(df)
# print(pd.get_dummies(df,prefix='', prefix_sep='').groupby(level=0, axis=1).max().set_index("year"))

# df_processed = pd.get_dummies(df, prefix_sep="_", columns=['color'])
# print(df_processed.head(10))
# df_processed.to_csv('df_processed_get_dummies.csv')


df = pd.DataFrame([
    [101, 'roof', 'garage', 'basement'],
    [102, 'basement', 'garage', 'painting'],
])
df.columns = ['no', 'point1', 'point2', 'point3']
print(df)

#print(pd.get_dummies(df))
print(pd.get_dummies(df,prefix='', prefix_sep='').groupby(level=0, axis=1).max().set_index("no"))
