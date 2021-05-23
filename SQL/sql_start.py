import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import select
import psycopg2

engine = create_engine('postgresql+psycopg2://postgres:37561239*@localhost:5432/northwind')
engine.connect()

print(engine)

sql = '''
select product_id, product_name, unit_price
from products p
where unit_price > 22
'''

df_sql = pd.read_sql(sql, engine)
print(df_sql.head())

# df = 'наш dataframe'

# df.to_sql('table_name_in_sql', con)  # отправить данные в таблицу
#
# sql = '''select * from table_name_in_sql t'''  # запрос
#
# df_sql = pd.read_sql(sql, con)  # отправка запроса в базу данных и получение данных (таблицы)



