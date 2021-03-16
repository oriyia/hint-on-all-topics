import psycopg2 as ps
import sqlalchemy as sq
import pandas as pd
import openpyxl
pd.set_option('max_colwidth', 800, 'display.max_columns', 10, 'display.width', 1000)


conn = 'postgresql+psycopg2://readonly:6hajV34RTQfmxhS@dsstudents.skillbox.ru:5432/db_ds_students'
engine = sq.create_engine(conn)  # отдельное подключение (engine стандарт названия)
connect = engine.connect()  # подключение
inspector = sq.inspect(engine)
table = inspector.get_table_names()

# df = pd.read_sql('select r.*, k.* '
#                  'from ratings r '
#                  'left join keywords k '
#                  'on r.movieid=k.movieid', connect).drop_duplicates('movieid', ax)
# print(df.head(5))
import json
df = pd.read_csv('keywords.csv')
print(df.head(5))
movieid_l = []
import re

import demjson
import ast
# a = ast.literal_eval(df.tags.iloc[0])
# c = f'{demjson.encode(a)}'
# print(c)
# b = json.loads(c)
# print(b)
# t = json.dumps(b)
# print(t)

# a = '{"id":5202,"name":"boy"}'
# a2 = "{'id': 931, 'name': 'jealousy'}"
# b = demjson.encode(demjson.decode(a2))
# print(type(b))
# print(b)



# for i, x in enumerate(df.tags):
#     x = x.replace('\'id\'', '\"id\"')
#     x = x.replace('\'name\'', '\"name\"')
#     x = x.replace('\\', '')
#     for match in re.findall(r':\s(\'[^}]*?\')}', x):
#         x = x.replace(match, f'\"{match[1:-1]}\"')
#     for match in re.findall(r'\s(\"[^"]*?\")\"', x):
#         x = x.replace(match, f'\'{match[1:-1]}\'')
#     try:
#         dd = json.loads(x)
#     except:
#         print('asdf')
#     for y in json.loads(x):
#         if y['name'] == 'africa':
#             movieid_l.append(df.movieid.iloc[i])
# movieid_l = set(movieid_l)
# print(len(movieid_l))
df_ratings = pd.read_sql('select * from ratings', connect)
df_keywords = pd.read_sql('select * from keywords', connect)

df_1 = df_ratings.merge(df_keywords, left_on='movieid', right_on='movieid', how='left').fillna(0)
print(df_ratings.shape, df_1.shape)
print(df_1.head(5))

for i, x in enumerate(df_1.tags):
    if x == 0:
        continue
    x = ast.literal_eval(x)
    for y in x:
        if y['name'] == 'africa':
            if df_1.movieid.iloc[i] not in movieid_l:
                movieid_l.append(df_1.movieid.iloc[i])
print(movieid_l)
print(len(movieid_l))

