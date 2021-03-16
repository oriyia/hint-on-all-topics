import psycopg2 as ps

con = ps.connect(database='postgres',
                 user='postgres',
                 password='37561239*',
                 host='127.0.0.1',
                 port='5432')
print('успех')