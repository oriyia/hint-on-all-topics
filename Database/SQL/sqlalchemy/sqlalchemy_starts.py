# ORM - технология сопоставления моделей, типы которых несопоставимы (классы с БД) (с самого начала
# развязать объектную модель и схему БД)
# каждый класс отображается на таблицу в базе данных
# не зарывайся, с помощью алхимии все не получится сделать. Дизайнить базу все равно придется.
# не надо пытаться натянуть алхимию на все



# from sqlalchemy.ext.declarative import declarative_base  # связка с базой данных
# base = declarative_base()  # прячет кишки классического маппинга под капот.
import psycopg2
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
# объект с данными
# class User(base):  # наследует базу
#     __tablename__ = 'users'  # яркий пример, насколько декларативна не была бы, дизайнить все равно надо
#     id = Column(Integer, primary_key=True)  # id важен для связи
#     name = Column(String)
#     fullname = Column(String)
#     def __repr__(self):
#         return '<User(name="{}", fullname"{}")>'.format(self.name, self.fullname)  #вывод

# a = base.metadata.create_all(engine)  # metadata указыает какие столбца и типы даннных в них лежат. А также мы
# просим создай в engine табличку

engine = create_engine('postgres+psycopg2://postgres:37561239*@localhost/postgres', echo=True)  # выбор базы данных с который будем работать
# и название базы, echo - полезный инструмент для отладки (показывает какие запросы делает алхимия) 5432
# Класс Engine связывает пул и диалект вместе, обеспечивая источник подключения и поведения базы данных.
# (mysql+pymysql://<username>:<password>@<host>/<dbname>)
# engine = create_engine("mysql://user:pwd@localhost/college",echo = True)

meta = MetaData()  # ~ коллекция объектов table
students = Table('students', meta,
                 Column('id', Integer, primary_key=True),
                 Column('name', String),
                 Column('lastname', String),)

meta.create_all(engine)  # функция create_all исп-ет объект engine для созд таблицы и сохр-ет в метаданных

# добавление строки в таблицу
# ins = students.insert().values(name='asdf', lastname='DFasfdfdsfs')
conn = engine.connect()  # объект соединения
# result = conn.execute(ins)


# Заполнение таблицы, сразу много
# result = conn.execute(students.insert(),
#                       [{'name': 'asdsdf', 'lastname': 'asdfdsxcc'},
#                        {'name': 'asdfdsfcxv', 'lastname': 'asdfdcxxcs'},
#                        {'name': 'asxcv', 'lastname': 'asdxcvfds'},
#                        {'name': 'asxz', 'lastname': 'asdfxcvxds'},])


# вывод всей таблицы
# s = students.select()  # объект select
# conn = engine.connect()
# result = conn.execute(s)  # используем как параметр для метода execute
# for row in result:
#    print (row)

# использование WHERE - УСЛОВИЕ
# s = students.select().where(students.c.id>2)  # c - атрибут, который являетя псевдонимом столбца
# result = conn.execute(s)
# for row in result:
#    print (row)


# Использование текстового SQL
from sqlalchemy.sql import text

# вывод всей таблицы
# t = text('select * from students')
# result = conn.execute(t)
# for row in result:
#    print (row)

# вывод имен, которые начинаются с А до L
# s = text("select students.name, students.lastname from students where students.name between :x and :y")
# result = conn.execute(s, x = 'A', y = 'L').fetchall()

# Условие
from sqlalchemy.sql import select
# s = select([text('students.name, students.lastname from students')]).where(text('students.name between :x and :y'))
# conn.execute(s, x='A', y='L').fetchall()

# сразу несколько условий and_
from sqlalchemy.sql import and_
# s = select([text('students.name, students.lastname from students')])\
#     .where(and_(text('students.name between :x and :y'),
#                 text('students.id > 3')))
# conn.execute(s, x='A', y='L')

# ПСЕВДОНИМ
from sqlalchemy.sql import alias
# st = students.alias('a')
# s = select([st]).where(text(st.c.id > 2)) почему тут "с" есть хз
# это равнозначно SELECT a.id, a.name, a.lastname FROM students AS a WHERE a.id > 2
# conn.execute(s)

# UPDATE
# s = students.update().where(students.c.lastname == 'asdfxcvxds').values(lastname = 'Ярусов')
# conn.execute(s)

# DELETE
# s = students.delete().where(students.c.id > 2)

# создание новой таблицы
from sqlalchemy import ForeignKey
addresses = Table('addresses', meta,
                 Column('id', Integer, primary_key=True),
                 Column('st_id', Integer, ForeignKey('students.id')),
                 Column('postal_add', String),
                 Column('email_add', String))

meta.create_all(engine)
# заполнение
# conn.execute(addresses.insert(),
#              [{'st_id':1, 'postal_add':'Shivajinagar Pune', 'email_add':'ravi@gmail.com'},
#               {'st_id': 1, 'postal_add': 'ChurchGate Mumbai', 'email_add': 'kapoor@gmail.com'},
#               {'st_id': 3, 'postal_add': 'Jubilee Hills Hyderabad', 'email_add': 'komal@gmail.com'},
#               {'st_id': 5, 'postal_add': 'MG Road Bangaluru', 'email_add': 'as@yahoo.com'},
#               {'st_id': 2, 'postal_add': 'Cannought Place new Delhi', 'email_add': 'admin@khanna.com'},])

# вывод сразу две совмещенные таблицы
# s = select([students, addresses]).where(students.c.id == addresses.c.st_id)
# result = conn.execute(s)

# UPDATE СРАЗУ НЕСКОЛЬКО ТАБЛИЦ (ДОСТУПНО ТОЛЬКО POSTGRES AND MICROSOFT)
# s = students.update().values({students.c.name:'xyz', addresses.c.email_add:'abc@xyz.com'})\
#     .where(students.c.id==addresses.c.id) # тут почему-то ошибка
# conn.execute(s)


# СОЕДИНЕНИЕ
# join(right, onclause=None, isouter=False, full=False) - right - правая сторона объединения,
# onclause — выражение SQL, представляющее предложение ON объединения. Если оставить None,
# он попытается объединить две таблицы на основе отношения внешнего ключа
# isouter — если True, то рендерится ЛЕВОЕ ВНЕШНЕЕ СОЕДИНЕНИЕ, а не СОЕДИНЕНИЕ
# full — если True, рендерит FULL OUTER JOIN вместо LEFT OUTER JOIN
# s = students.join(addresses, students.c.id==addresses.c.st_id)
# stmt = select([students]).select_from(s)
# r = conn.execute(stmt)
# print(r.fetchall())


# КОНЪЮНКЦИЯ
# and_
# s = select([text('students.name, students.lastname from students')])\
#     .where(and_(text('students.name between :x and :y'),
#                 text('students.id > 3')))
# or_
# from sqlalchemy.sql import or_
# s = select([students]).where(or_(students.c.name=='asdf', students.c.id>2))
# conn.execute(s)
# print(s.fetchall())

# СОРТИРОВКА
# ASC() по возрастанию
from sqlalchemy import asc
# s = select([students]).order_by(asc(students.c.name))
# аналогично и DESC() - по убыванию

# BETWEEN() - выбирает промежуток
from sqlalchemy import between
# s = select([students]).where(between(students.c.id,3,6))  # 3 и 6 тоже включаются
# r = conn.execute(s)


# ИСПОЛЬЗОВАНИЕ ФУНКЦИЙ
# NOW()
from sqlalchemy import func
# r = conn.execute(select([func.now()]))  # вывод даты на данный момент
# print(r.fetchone())

# COUNT()
# r = conn.execute(select([func.count(students.c.id)]))  # количество строк
# print(r.fetchone())
# также реализованы и max, min, avg,


# ОПЕРАЦИИ НАД МНОЖЕСТВАМИ
# UNION
from sqlalchemy import union
# Объединяя результаты двух или более операторов SELECT, UNION удаляет дубликаты из набора результатов
# но тут что-то не так, ошибка
# s = union(students.select().where(addresses.c.email_add.like(%@gmail.com addresses.select().where(addresses.c.email_add.like('%@yahoo.com')))))

# UNION_ALL
from sqlalchemy import union_all
# Операция UNION ALL не может удалить дубликаты и не может отсортировать данные в наборе результатов.
# u = union_all(addresses.select().where(addresses.c.email_add.like('%@gmail.com')), addresses.select().where(addresses.c.email_add.like('%@yahoo.com')))

# EXCEPT_
from sqlalchemy import except_
# Предложение / оператор SQL EXCEPT используется для объединения двух операторов SELECT и возврата строк из
# первого оператора SELECT, которые не возвращаются вторым оператором SELECT. Функция exc_ () генерирует
# выражение SELECT с предложением EXCEPT.
# В следующем примере функция exc_ () возвращает только те записи из таблицы адресов, которые имеют «gmail.com»
# в поле email_add, но исключают те, которые имеют «Pune» как часть поля postal_add.

# u = except_(addresses.select().where(addresses.c.email_add.like('%@gmail.com')), addresses.select().where(addresses.c.postal_add.like('%Pune')))


# INTERSECT()
from sqlalchemy import intersect
# В следующих примерах две конструкции SELECT являются параметрами для функции intersect ().
# Одна возвращает строки, содержащие «gmail.com», как часть столбца email_add, а другая возвращает строки,
# содержащие «Pune» как часть столбца postal_add. Результатом будут общие строки из обоих наборов результатов.
# u = intersect(addresses.select().where(addresses.c.email_add.like('%@gmail.com')), addresses.select().where(addresses.c.postal_add.like('%Pune')))


