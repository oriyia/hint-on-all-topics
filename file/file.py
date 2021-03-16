import pandas as pd
import plotly.offline as offline
import plotly.express as px
import json
# f = open('text.txt', 'w')
# или open(r'C:\Python33\Lib\pdf.py') -- любые косые черты: \, /, \\
# f.write('hello\n')
# f.write('world\n')
# f.close()

# f = open('text.txt')
# text = f.read()
# print(text)

# for line in open('text.txt'): print(line, end='')

# dir(f) список всех методов для этого файла
# help(f.метод) что делает метод для этого файла

# data = open('text.txt', 'rb').read()

# line = f.readline() -- чтение одной строки
# line.rstrip() -- удаление символа конца строки \n


# МОДУЛЬ PICKLE - бинарный метод хранения данных
# import pickle
# f = open('datafile.pkl', 'wb')
# pickle.dump(object, f)
# # для воссоздания
# f = open('datafile.pkl', 'rb')
# e = pickle.load(f)




# lines = [line.rstrip() for line in open('file.py')]
# [line.rstrip().upper() for line in open('file.py')]



# CSV - файл данных с разделителями (по умолчанию запятая).
# df = pd.read_csv('название файла, или путь', sep=';', index_col='название столбца индекса', nrows=1)  # загрузка
# количества строк начинаю с первой
# df.to_csv('Название нового файла', index_label='название столбца с индексами', sep=';')  # сохранение
# portfel.to_csv('tr.csv', index=False, mode='a', header=False) пересохранение с добавлением, без учета индекса пандаса
# , вместо mode='w' будет 'a', и не учитываем строки заголовка пандасa

# РАБОТА С ДАТАМИ
# СТАНДАРТ ЗАПИСИ - 2020-02-29
# преобразование столбца в формат даты пандаса
# pd.read_csv('file.csv', parse_dates=['название столбца с датами'], dayfirst=False)
# dayfirst=True если день левее месяца, иначе False

# df['Дни недели'] = df['Date'].dt.day_time()  # создание столбца с днями недели
# df.['Недели'] = df['Date'].dt.month  # создание столбца с неделями



# XLSX
# df = pd.read_excel('file.xlsx', sheet_name='название страницы в excel')  # можно загрузить только одну страницу
# df.to_excel('название.xlsx', index_label='sadf')  # сохранение





# ПРАКТИКА
# df = pd.read_csv('Homework1.csv', index_col='id', parse_dates=['Date'], dayfirst=True)
# print(df.head())
# print(df['Temperature'].mean())

# df.to_csv('result1.csv', index_label='index')
# df.to_excel('result2.xlsx')

# dfd = pd.read_csv('Homework2.csv', sep=';', parse_dates=['Date'], dayfirst=True, index_col='id')
# print(dfd.head())

# dfd.to_excel('result3.xlsx', sheet_name='MySheet')



# df1 = pd.read_csv('Homework1.csv', parse_dates=['Date'], dayfirst=True)
# df4 = pd.read_csv('Homework4.csv')
# print(df1.head())
# print(df4.head())

# df1 = df1.merge(df4, on='id', how='left')
# print(df1.head())
# df1.to_csv('result4.csv', sep=';')



# df2 = pd.read_csv('Homework2.csv', sep=';', parse_dates=['Date'], dayfirst=True, index_col='id')
# df5 = pd.read_excel('Homework5.xlsx', parse_dates=['Date'])
# print(df2, df5)

# df25 = pd.merge(df2, df5, on='Date', how='outer')
# df25.to_csv('result5.csv', sep="\t", encoding='utf-8')

# df25['Temperature_C'] = (df25['Temperature'] - 32) * 5 / 9
# print(df25)
# df25.to_csv('result6.csv')

# fg = px.scatter(df25, x=df25.Temperature_C, y=df25.Sales, color=df25.Temperature_C)
# offline.plot(fg)


# ЧТЕНИЕ ДАННЫХ ИЗ ВЕБ-СЕРВИСА

# МОДУЛЬ JSON
# раньше был только в javascript
# легкий и доступны обмен информацией между сервером и приложением, между разными программами, языками програ-ия
#
# ОДИНОЧНЫЕ ' НЕ ЯВЛЯЮТСЯ JSON, НУЖНО ДВОЙНЫЕ
# str = str.replace("\'", "\"")

import demjson
import ast
# a = ast.literal_eval(df.tags.iloc[0]) -- "{'id':5202,'name':'boy'}" убирает внешние ковычки {'id':5202,'name':'boy'}
# c = f'{demjson.encode(a)}' - преобразовывает внутри dict одинарные в двойные
# print(c)
# b = json.loads(c) - загрузка json, формат json --> в формат python
# print(b)
# t = json.dumps(b) - формат python --> в формат json
# print(t)
# b = demjson.encode(demjson.decode(a2)) --можно так, убираем кавычки, меняем внутри ' на ", + меняется снаружи на '


# import json
# json.dump(object, fp=open('textfile.txt', 'w'), indent=4) - сохранение json в файл (object - то что хотим сохранить
# fp - туда куда хотим, т.е. файл на компе, indent=4 нужне для красивого форматирования (лучше читается глазами)
# после надо закрыть файл fp, чтобы можно было дальше его использовать
# поэтому лучше так
# with open('textfile.json', 'w') as file:
#     json.dump(object, file, indent=4)
# print(open('textfile.txt').read())
# p = json.load(open('textfile.txt'))
# для открытия
# with open('textfile.json', 'r') as file:
#     a = json.load(file)
import urllib.request as req
# url = 'https://www.metaweather.com/api/location/search/?query=moscow'
# session = req.urlopen(url)  # отправка запроса на сервер
# !!! ВАЖНО ЗАКРЫВАТЬ СОЕДИНЕНИЕ, ИНАЧЕ ПРОБЛЕМЫ С ИНЕТОМ
# with req.urlopen(url) as session:  # использование менеджера контекста
#     response = session.read().decode()
# print(response)
# data = json.loads(response)
# print(data[0])  # вывод первого элемента в jsone
# city = data[0]['woeid']
# print(city)
# url2 = 'https://www.metaweather.com/api/location/2122265/2020/06/20/'
# with req.urlopen(url2) as session2:  # использование менеджера контекста
#     response2 = session2.read().decode()
# print(response)
# df = pd.read_json(response2)
# df = pd.read_json('https://www.metaweather.com/api/location/2122265/2020/06/20/') # можно и так
# df = pd.read_json('file.json')


url = 'https://www.metaweather.com/api/location/search/?query=london'
with req.urlopen(url) as session:
    response = session.read().decode()
data = json.loads(response)
print(data[0]['woeid'])

url2 = 'https://www.metaweather.com/api/location/44418/2019/01/03/'
with req.urlopen(url2) as session2:
    response2 = session2.read().decode()

pd.set_option('max_colwidth', 800, 'display.max_columns', 13, 'display.width', 1000)

df = pd.read_json(response2)
print(df['the_temp'].mean())

weather = json.loads(response2)
json.dump(weather, open('wether.json', 'w'), indent=4)

data_weather = pd.read_json('wether.json')
print(data_weather.head(5))


data_weather.to_csv('result.csv', index_label='index')

import psycopg2