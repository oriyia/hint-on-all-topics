import pymongo as pm
import pandas as pd
import numpy as np

import string

df1 = pd.DataFrame([[-1, 2, 3.2],
                   [-4, 5, 6.1],
                   [-7, 8, 7.7]], columns=['a', 'b', 'c'])
df2 = pd.DataFrame([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]], columns=['a', 'g', 'x'])
def maxx(x):
    return x['a'].min() > 6


df_int = df1.select_dtypes(include=['float'])
convert = df_int.apply(pd.to_numeric, downcast='float')
print(df1)
print(df_int)
print(convert)
print(df1.dtypes)
print(convert.dtypes)

def mem_useg(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2
    return "{:03.19f} MB".format(usage_mb)

print(mem_useg(df1['c']))
print(mem_useg(convert))

a = df1['c'].value_counts()
b = df1['c'].value_counts().values
c = df1['c'].value_counts()[df1['c']].values
print(a)
print(type(b))
print(type(c))

