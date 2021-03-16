import psycopg2 as ps
import sqlalchemy as sq
import pandas as pd
import openpyxl

conn = 'postgresql+psycopg2://readonly:6hajV34RTQfmxhS@dsstudents.skillbox.ru:5432/db_ds_students'# в строке указыается: язык sql, драйвер подключения, имя пользователя: readonly, пароль - 6hajV34RTQfmxhS,
# адрес субд - @dsstudents.skillbox.ru, порт, и название базы данных

engine = sq.create_engine(conn)  # отдельное подключение (engine стандарт названия)
connect = engine.connect()  # подключение
inspector = sq.inspect(engine)
table = inspector.get_table_names()
print(table)

# df = pd.read_sql('select * from course_purchases', connect).iloc[1:,:]

df = pd.read_sql('select * from ratings', connect)
print(df.head(5))
df1 = df[:100]
print(df1.head(5))
df[:10000].to_excel('result2.xlsx', sheet_name='MySheet')