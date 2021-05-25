import pandas as pd
import numpy as np
import datetime

# НАСТРОЙКИ ОТОБРАЖЕНИЯ
# pd.options.display.max_rows = 1000  # максимальное число выводимых строк
# pd.set_option('max_colwidth', 800)  максимальное отображаемое количество символов в строке
# pd.set_option('display.width', 900)  ширина дисплея вывода
# pd.set_option('display.max_columns', 14)  # максимальное количество отображаемых столбцов
# pd.set_option('max_colwidth', 800, 'display.max_columns', 10, 'display.width', 1000)


# ОБЪЕКТ Series
# аргументы data и index(необязательный)
# объект Series — аналог одномерного массива с гибкими индексами, в нем значения одного типа


# СОЗДАНИЕ
# data = pd.Series([0, 0.1, 0.2, 0.3]) передача python списка
# pd.Series([2]*5) - [2, 2, 2, 2, 2]
# pd.Series(list('abcd')) - a, b, c, d
# pd.Series({'Ключи индекса': 'значения'}) передача словаря
# pd.Series(np.arange(3, 6)) 3, 4, 5 -  передача массива numpy (6 не входит)
# использование скалярного выражения
# print(data.values)  # array[0.  0.1 0.2 0.3] - объект numpy
# print(data.index)  # RangeIndex(start=0, stop=4, step=1) - стандартный индекс в pandas
# print(data.index.values)  - вывод списка со значениями индекса
# data = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
# data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[2, 5, 3, 7])
# print(list(data.keys()))  # [2, 5, 3, 7]
# print(list(data.items()))  # [(2, 0.25), (5, 0.5), (3, 0.75), (7, 1.0)]


# СРЕЗЫ - не создают новых объектов, нужно пересохранять
# явный индекс - data['a':'c'] "с" включается, в отличии от неявного data[0:2]
# print(data[2:8:2])  # 3 не будет включаться, ШАГ 2
# data[::-1] - в обратном порядке
# data[::-2] - в обратном порядке с шагом 2
# data[:-2] - вывод всех кроме двух последних
# data['a':'c'] - "c" уже включается в вывод
# # ИНДЕКСАТОРЫ loc, iloc - поиск осуществлять только ими ([], .ix[] - нельзя)
# # print(data.loc[2])  # 0.25 явный индекс
# # print(data.iloc[2])  # 0.75  неявный индекс
# подобное преобразует выбранную строку в объект Series, где индексами становятся названия столбцов


# ОБЪЕКТ DataFrame — аналог двумерного массива с гибкими индексами строк и гибкими именами столбцов
# population = pd.Series({'Калифорния:': 34000,
#                         'Москва': 35000,
#                         'Питер': 45933,
#                         'Казань': 23012})
# area = pd.Series({'Калифорния:': 1234500,
#                   'Москва': 3545000,
#                   'Питер': 459334353,
#                   'Казань': 230123534})
# table = pd.DataFrame({'Популяция': population, 'Население': area})
# print(table.index)  # Index(['Калифорния', 'Москва', 'Питер', 'Казань'], dtype='object')
# print(table.index.values)  # ['Калифорния', 'Москва', 'Питер', 'Казань']
# print(table.columns)  # Index(['Популяция', 'Население'], dtype='object')


# # СПОСОБЫ СОЗДАНИЯ
# результатом функции Numpy
# a = pd.DataFrame(np.random.rand(3, 2), columns=['foo', 'bar'], index=['a', 'b', 'c'])
# a = pd.DataFrame([['2019-10-13 00:00:00', 0]], columns=['date', 'price']) 1 строка, 2 столбца
# загрузка npy
# arr_pandas = np.load('arr_pandas.npy', allow_pickle=True)
# arr = pd.DataFrame(arr_pandas)
# словарем со списками
# pd.DateFrame({'название столбца': [можно и одно значение, но список]})
# с помощью объектов Series
# pd.DataFrame([Series1, Series2])
# СОЗДАЕМ СТОЛБЦЫ
# table['new_col'] = [...] или Series например table['col'] / 2
# table.insert(1, 'name_col', Series) - вставка столбца в определенную позицию
# table[:, 'new_col'] = 0 - добавление столбца со всеми нулевыми значениями

# -------------------------
# ИЗМЕНЕНИЕ ТАБЛИЦ
# ПЕРЕИМЕНОВАНИЕ СТОЛБЦОВ
# table.rename(columns={'что хотим изм': 'на что хотим заменить'}, inplace=True)
# ИЛИ
# arr.columns = ['id', 'dist', 'climb', 'time', 'timef', 'type']
# a.columns = []  a.index = []
# ДОБАВЛЕНИЕ СТРОК APPEND
# table.append(table2, ignore_index=True) - если были разные столбцы, то заполнится значениями Nan
# ignore_index - не сохраняет предыдущих индексов


# ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ DATAFRAME
# ВЫВОД СТОЛБЦОВ
# # table['Population']  # вывод столбца или (результат объект Series)
# подобное преобразует выбранную строку в объект Series, где индексами становятся названия столбцов
# table.Population  если, конечно, нет пробелов
# # print(table[['Популяция', 'Население']])  # вывод 2-х столбцов (результат объект DataFrame)
# ВЫВОД СТРОК
# table[2:4] - вывод 2 и 3 строк
# table.loc['aa'] - вывод строки по метке индекса (выводится объект Series)
# table.loc[['aa']] - вывод строки (выводится будет уже объект DataFrame)
# table.loc[['aa', 'bb']] - вывод несколько строк по списку меток индекса
# table.iloc[1] - вывод второй строки по позиции (выводится объект Series)
# table.iloc[[1]] - вывод второй строки по позиции (выводится объект DataFrame)
# table.iloc[[1, 2]] - вывод несколько строк по позиции (DataFrame)
# ВЫВОД СРАЗУ ПО СТРОКАМ И СТОЛБЦАМ
# table[['aa', 'bb']][['cc', 'dd']] - вывод 2-х строк и 2-х столбцов
# СРЕЗЫ
# # table.iloc[:3, :2] равно table.loc[:'Питер', :'area']
# # table.iloc[0, 2] = 20
# # print(table.loc['Питер', 'Население']) вывод одного значения
# # print(table.loc[:'Питер', :])
# # print(table.loc[:'Питер', ['Популяция', 'Отношение']])  вывод только двух столбцов и всех строк до Питера
# # print(table.head(2))  # вывод только первые две строки
# # print(table.tail(2))  # вывод последних двух строк
# # print(table.loc[список с индексами])
# ВНИМАТЕЛЬНЕЕ СО СРЕЗАМИ
# aa = df[:5] - аа ссылка на фрейм df, если изменить аа, то изменить и df. Нужно тогда взять копию
# aa = df[:5].copy()


# УСЛОВИЕ (логические отбор - он мощнее даже чем в SQL where)
# # indA & indB # пересечение (И)
# # indA | indB # объединение (ИЛИ)
# # indA ^ indB # симметричная разность
# table[table.Population > 2] - выводится попадая под условие только то что внутри будет ТРУ
# table.loc[table['Отношение'] > 0.001, ['Население', 'Отношение']]  # вывод 2 столбцов с выше 0.001
# table[table.Population > 2].all() - все ли элементы True
# table[table.Population > 2].any() - есть ли хоть элемент True (т.е. больше 2-х)
# table[table.Population > 2].sum() - сумма элементов больших 2-х
# table[(~table.Population.isin([2, 3])) & (и что-то еще)] - условие где не равно 2, 3


# ИНФОРМАЦИЯ ПО DATAFRAME
# # print(table.info)
# # print(table.describe())  # вывод инфы по каждому столбцу (кол, сред, мин, макс ...)
# для нечисловых данных другие метрики - кол-во, частота, самое часто встречаемое и ...


# # ОБЪЕКТ INDEX - с индексами быстрее осуществляется поиск
# если индекс не задается при создании, то стандарт RangeIndex
# 1) RangeIndex - стандартный индекс, раньше Int64Index
# # ind = pd.Index([2, 3, 4, 5])  # Int64Index([2, 3, 4, 5], dtype='int64')
# 2) Float64Index - числа с плавающей запятой
# DateTimeIndex
# если индекс объект времени то индекс объект DateTimeIndex
# infer_objects()  # определение типа столбцов
# ПЕРЕИНДЕКСАЦИЯ
# df.set_index('col') - определение столбца как новый индекс
# s2 = s1.reindex(index=[1, 2, 3, 4]) - данные из s1 копируются по меткам индекса и вставляются в s2, если не найдено,
# то будет Nan, если в s1 были метки не участвовавшие, то они удаляются
# s2 = s1.reindex(index=[1, 2, 3, 4], fill_value=0)
# но можно и изменить просто индексы
# s1.reindex([новые индексы], method='ffill') если индексов будет больше, то Nan заменятся на значения
# из предыдущей строки ('bfill' - обратное)
# s1.reindex(columns=['aa', 'vv']) - переиндексация для столбцов
# Номер позиции строки
# m = s1.index.get_loc('aa') - поиск позиции метки s1.iloc[m]
# СБРОС ИНДЕКСА
# table.reset_index() - значения индекса перемещаются в столбец
# table['aa'].set_index() - перемещение столбца в индекс


# # ЗАПОЛНЕНИЕ ДЛЯ НЕДОСТАЮЩИХ ЭЛЕМЕНТОВ
# # A.add(B, fill_value=0)  недостающий элемент будет равен нулю


# # ОПЕРАЦИИ НАД МАССИВАМИ - но если какие-то метки индекса или столбцов не найдены в том или другом датафрейме
# то будет результатный фрейм будет со всеми метками но там будет Nan
# # + add()
# # – table.sub(вычитаемое, axis=0), subtract()
# # * mul(), multiply()
# # / truediv(), div(), divide()
# # // floordiv()
# # % mod()
# # * pow()
# УНИКАЛЬНЫЙ ЗНАЧЕНИЯ
# table.unique() - возвращает список уникальных значений
# КОЛИЧЕСТВО УНИКАЛЬНЫХ ЗНАЧЕНИЙ - Nan не включаеются, чтобы вкл, то dropna=False
# table.nunique(dropna=False)
# ВСТРЕЧАЕМОСТЬ КАЖДОГО УНИКАЛЬНОГО ЗНАЧЕНИЯ
# table.value_counts(dropna=False)
# table.value_counts(normalize=True) - подсчет количества, но вывод в отношении
# table.value_counts(sort=True, ascending=True) - подсчет с сортировкой по убыванию
# table.value_counts(bins=4) - разбиение на 4 равные группы и подсчет количества в каждой группе
# ПОЗИЦИЯ ИНДЕКСА
# table['aa'].idxmax() - для максимального значения, так же и min
# НАХОЖДЕНИЕ N-НАИМЕНЬШИХ И НАИБОЛЬШИХ ЗНАЧЕНИЙ
# table.nsmallest(4, ['aa'])['aa'] - 4 наимешьних значений в столбце 'aa' (nlargest - для наибольших)
# ВЫЧИСЛЕНИЕ НАКОПЛЕННЫХ ЗНАЧЕНИЙ
# pd.Series([1, 2, 3, 4]).cumprod() - накопленное произведение 1, 2, 6, 24
# a=pd.Series([1, 2, 3, 4]).cumsum() - накопленная сумма 1, 3, 6, 10
# МОДА, МЕДИАНА, СРЕДНЕЕ, ДИСПЕРСИЯ, СТАНДАРТНОЕ ОТКЛОНЕНИЕ
# table.mean(axis=1) - среднее значение для каждой строки
# table.mode() - мода- то что чаще встречается (может быть несколько)
# table.var() - дисперсия
# table.std() - стандартное отклонение
# КОВАРИАЦИЯ И КОРРЕЛЯЦИЯ
# table.cov() - ковариация + -
# table.corr() - корреляция от -1 до +1
# ДИСКРЕТИЗАЦИЯ И КВАНТИЛИЗАЦИЯ
# bins = pd.cut(фрейм, 4) - разбить фрейм на 4 группы
# разбивка на возрастные группы
# group_age = pd.cut([список с возрастами], [6, 18, 25, 35, 75], labels=[название для групп])
# pd.qcut(фрейм, 5) - разбивка фрейма на 5 одинаковых групп
# РАНЖИРОВАНИЕ
# table.rank() - выставление порядкового номера от меньшего к большему
# ПРОЦЕНТНОЕ ИЗМЕНЕНИЕ ДЛЯ КАЖДОЙ СТРОКИ
# table['aa'].pct_change()
# СКОЛЬЗЯЩЕЕ rolling
# s = s.rolling(windows=3).mean() - скользящее окно шириной 3 среднее
# s = s.rolling(windows=3).apply(np.mean()) - тоже самое
# table.rolling_mean() - скользящее среднее с расширяющимся окном
# means = s.mean() - скользящее среднее по трем дня по всему столбцу
# СЛУЧАЙНАЯ ВЫБОРКА ДАННЫХ
# table.sample(n=3) - выбор 3-х строк
# table.sample(frac=0.1, replace=True) - выбор 10% случайных данных с возвращением данных
# ЗАМЕНА ЗНАЧЕНИЙ
# table.replase([1, 2, 3, 4], [3, 4, 5, 8]) - сразу несколько
# table.replase({0: 2, 1: 100}) - замена по ключу
# table.replase({'a': 1, 'b': 3}, 100) - замена значения к каждом столбце на 100


# УДАЛЕНИЕ ДАННЫХ
# table.drop(['col'], axis=1) - удаление столбца
# table.drop(['row1', 'row2'], axis=0) - удаление строк
# УДАЛЕНИЕ ДУПЛИКАТОВ
# table.duplicates() - показывает дупликаты True/False
# table.drop_duplicates() - удаляет дупликаты, из дупликатов остается первая встретившаяся строка
# table.drop_duplicates(keep='last') - наоборот остается последняя
# table.drop_duplicates(['aa', 'bb']) - по этим столбцам поиск дупликатов строк


# # СОРТИРОВКА
# # table.sort_values('столбец')  # не изменяет исходный массив
# # table.sort_values(['1столбец', '2столбец'], ascending=False)  # сразу по 2-м столбцам по убыванию
# # table.sort_index()


# # РАБОТА С ПРОПУЩЕННЫМИ ДАННЫМИ
# # значение - индикатор Nan - значение с плавающей точкой
# # Nan превращает None в Nan
# ПОИСК ПРОПУЩЕННЫХ
# table.isnull().sum()  # подсчет количества незаполненых элементов по каждому столбцу
# table.notnull().sum(axis=1)  # подсчет количества заполненых элементов по каждой строке
# ЗАПОЛНЕНИЕ ПРОПУЩЕННЫХ
# # table.fillna(0)  # заполнение элементов Nan нулями
# # table.fillna(method='ffill') копия предыдущей ячейки в ячейку Nan (в прямом порядке) = pd.ffill()
# # table.fillna(method='bfill', axis=1) копия следующей ячейки в яч Nan (в обратном порядке)= pd.bfill()
# УДАЛЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
# # table.dropna()  # удаление строк в которых есть Nan, НО нужно пересохранять
# # table.dropna(axis=1)  # удаление столбцов, в которых есть Nan
# # table.dropna(how='all' или 'any')  # удаление где хотя бы один Nan, all - только где все Nan
# # table.dropna(thresh=2)   # удалит если пропущенных 2 и более
# проверка пропусков по конкретному столбцу
# # students_birthday = students.dropna(subset=['birthday']).reset_index(drop=True)  # не сохраняет старый индекс
# # в новом столбце
# ИНТЕРПОЛЯЦИЯ
# table.interpolate()
# table.interpolate(method='time) - для столбца с датами


# КАТЕГОРИАЛЬНЫЕ ПЕРЕМЕННЫЕ
# df['col'].astype('category') - изменение типа столбца на категорию
# cat_val = pd.Categorical(['low', 'high', 'medium', 'low', 'high', 'medium'])
# определяет какое есть уникальное значение есть в списке и использует его в качестве категории
# print(cat_val) ['low','hight','medium','low','hight','medium']Categories (3, object): ['hight', 'low', 'medium']
# cat_val.categories - просмотр какие есть категории
# cat_val.get_values() - получение значений категории
# у каждой категории присваивается код
# print(cat_val.codes)  [1 0 2 1 0 2]
# явно указать категории
# cat_val_av = pd.Categorical(cat_val, categories=['low', 'medium', 'high'])
# СОРТИРОВКА - производится по кодам
# print(cat_val_av.sort_values()) - ['low', 'low', 'medium', 'medium', 'high', 'high']
# СОЗДАНИЕ Series как категориальную переменную
# ss = pd.Series(cat_val_av, dtype='category')
# СОЗДАНИЕ КАТЕГОРИАЛЬНОЙ ПЕРЕМЕННОЙ ЧЕРЕЗ ИЗМЕНЕНИЕ ТИПА
# aa.astype('category')
# .CAT - доступ к свойствам категориального Series
# print(ss.cat.categories) - Index(['low', 'medium', 'high'], dtype='object')
# ЯВНЫЙ ПОРЯДОК КАТЕГОРИЙ
# categ = ['low', 'medium', 'high']
# print(pd.Categorical(cat_val, categories=categ, ordered=True)) - Categories (3, object): ['low' < 'medium' < 'high']
#  ПЕРЕИМЕНОВАНИЕ КАТЕГОРИЙ
# cat_val.rename_categories(['x', 'y', 'z'])
# d = {'M': 'Male', 'F': 'Female'}
# cat_val['col'] = cat_vol['col'].map(d) - передаем словарь ключи старые имена категорий, значения - новые
# ДОБАВЛЕНИЕ КАТЕГОРИЙ
# aa = cat_val.add_categories(['veryster', 'asdf]) - Categories (4, object): ['high', 'low', 'medium', 'veryster']
# УДАЛЕНИЕ КАТЕГОРИЙ
# aa = cat_val.remove_categories(['high']) Categories (2, object): ['low', 'medium'] удаленное замен-ся на Nan
# УДАЛЕНИЕ НЕИСПОЛЬЗУЕМЫХ КАТЕГОРИЙ
# cat_val.remove_unused_categories()
# УСТАНОВКА КАТЕГОРИЙ
# ss = ss.cat.set_categories(['adsf', 'asdf'])


# # ИЕРАРХИЧЕСКОЕ ИНДЕКСИРОВАНИЕ - или мультииндексирование MultiIndex
# # table.set_index(['1столбец', '2столбец'], inplace=True) #новый индексы через два столбца, с пересохранкой
# # table.loc['1столбец', '2столбец']  # вывод значения
# table.index.levels(0) - вывод индексов первого уровня
# table.index.get_level_values(0) - вывод значений индекса с первого уровня
# XS - с его помощью можно получать только значения, а не устанавливать их
# table.xs('метка', level=1, drop_level=False) - вывод значений по метке индекса второго уровня, drop_level -
# чтобы не удалять индекс того уровня по которому производим отбор
# table.xs('sadf', level=0).xs('sdf', level=1) - отбор по нескольким уровням индекса ИЛИ
# table.xs(('dfsa', 'asdf')) - так тоже работает (в виде кортежа)


# # СПОСОБЫ СОЗДАНИЕ МУЛЬТИИНДЕКСА
# df = pd.DataFrame(np.random.rand(4, 2),
#                   index=[['a', 'a', 'b', 'b'], ['c', 'd', 'c', 'd']],
#                   columns=['data1', 'data2'])
# data = {('москва', 2000): 2342554254, ('москва', 2010): 2456762554254,
#         ('питер', 2000): 89889879, ('питер', 2010): 7646456667,
#         ('екб', 2000): 432564365, ('екб', 2010): 909090909090, }
# pd.Series(data)  # создание того же мультииндексного объекта
# pop.index.names = ['asdfd', 'dfasdfasdf']  # название уровней


# # МУЛЬТИИНДЕКС ДЛЯ СТОЛБЦОВ
# index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
#                                    names=['year', 'visit'])
# columns = pd.MultiIndex.from_product([['Bob', 'Alex', 'Sem'],
#                                       ['HR', 'Temp']],
#                                      names=['subject', 'type'])
# data = np.round(np.random.rand(4, 6), 1)
# health_date = pd.DataFrame(data, index=index, columns=columns)


# СРЕЗЫ МУЛЬТИИНДЕКСОВ
# health_date.loc[2013, 1]
# health_date[health_date > 0.5]
# health_date['Bob', 'HR']
# health_date.loc[2013, 2]
# health_date.loc[2013, 2]['Bob', 'HR'] - ура блять получилось
# health_date.loc[(2013, 2), ('Bob', 'HR')] - или так
# # idx = pd.IndexSlice  # спецфункция для создания срезов
# # print(health_date.loc[idx[:, 2], idx[:]])
# # НО ДЛЯ ВЫПОЛНЕНИЯ СРЕЗОВ НЕОБХОДИМО, ЧТОБЫ ИНДЕКСЫ БЫЛИ ОТСОРИРОВАННЫ ЛЕКСИКОГРАФИЧЕСКИ
# # data.sort_index()
# # data.sort_level()


# АГРЕГИРОВАНИЕ ДАННЫХ
# доступные операции: АГРЕГИРОВАНИЕ, ПРЕОБРАЗОВАНИЕ, ФИЛЬТРАЦИЯ

# # ОПЕРАЦИЯ GROPBY - группировка по группам
# df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
#                    'data': range(6), 'bla': range(6)}, columns=['key', 'data', 'bla'])
# df = df.groupby('key') - <pandas.core.groupby.generic.DataFrameGroupBy object at 0x01E95FA0>
# df = df.groupby(['key', 'key1']) - группировка по нескольким столбцам
# ГРУППИРОВКА МУЛЬТИИНДЕКСА
# multi.groupby(level=0) - по первому уровню мультииндекса
# multi.groupby(level=['lev1', 'lev2']) - по нескольким

# АГРЕГИРОВАНИЕ
# df.ngroups - количество групп
# df.groups - словарь с названием групп и индесами включенных строк
# df.size() - сводка о размерах каждой группы
# df.count() - количество элементов в каждом столбце
# df.get_group('name') - вывод опеределенной группы
# df.sum() - сумма элементов по ключу
# df.groupby('key')['bla'].sum() - для конкретного столбца
# МЕТОД aggregate() или короткая форма agg()
# df.groupby('key')['data'].aggregate(['sum', 'min'])) применение указанных методов к столбцу
# df.groupby('key').aggregate({'data': 'min',
#                              'bla': 'max'}))  # для каждого столбца свой метод
# df.groupby(['word']).aggregate({'doc_id': lambda m: m.count() / doc_count * 100})
# # ОБРАБОТКА СРАЗУ НЕСКОЛЬКИХ СТОЛБЦОВ ОДНИМ МЕТОДОМ
# df.groupby(['world', 'country'])[['col1', 'col2', 'col3', 'col4']].agg('mean')
# СТРОКИ ПРИ АГРЕГИРОВАНИИ
# df.groupby("content_id")['tag'].apply(lambda tags: ','.join(tags)) - строки через запятую
# grp = df.groupby('A').agg(B_sum=('B','sum'), C=('C', ', '.join)).reset_index()
# АГРЕГИРОВАНИЕ С УЧЕТОМ NULL
# df.groupby(['content_id'], dropna=False)['tag'].count()  # null будут учитываться


# # ПРЕОБРАЗОВАНИЕ
# TRANSFORM - применяется к каждому значению фрейма
# df1.transform(lambda x: x.fillna(np.mean(x))) - заполнение пропущенных значений средним
# df.groupby('key').transform(lambda x: x - x.mean()) - центрирование по каждому элементу
# # УНИКАЛЬНЫЕ ЗНАЧЕНИЕ
# # df.groupby('course_title')['module_title'].nunique().reset_index()  - количество моделй в каждом курсе. ресет
# # для вывода снова в датафрейме


# # ФИЛЬТРАЦИЯ
# # def maxx(x):
# #     return x['bla'].max() > 3
# функция отбирает из всех сформированных групп
# df.groupby('key').filter(maxx) - вывод групп только где max в bla больше 3
#


# # КОНКАТЕНАЦИЯ CONCAT()
# если в конкатенируемых объектах имеются разные столбца или строки, то отсутствующие элементы будут Nan
# Nan только по выравниванию
# # pd.concat([DataFrame1, DataFrame2], axis=0, join='outer'('inner'), ignore_index=False,
# #            keys=None, levels=None, names=None, verify_integrity=False, copy=True)
# axis=0 конкатенации по оси строк, а выравнивание по меткам столбцов, если по меткам индекса или столбца будет
# несовпадение то будет Nan
# keys=['d1', 'd2'] - если есть одинаковые метки индексов, то даем имена этим группам (типо мультииндекс)
# keys - не работает, если есть ignore_index=True
# отбор данных если конкат был по оси столбцов df3.loc[:]['d1', 'a'] или df3.d1.a
# join='inner'(по умол. 'outer') - остается только то, где совпадают метки выравнивания
# ignore_index=True - избавиться от дублирование меток в итоговом индексе


# # СЛИЯНИЕ И СОЕДИНЕНИЕ MERGE()
# # pd.merge(obj1, obj2, on=['col1', 'col2'], how='inner'(по умол), suffixes = ['_R', '_T']).drop('col', axis=1)
# # ключевое слово on (работает, если в обоих объектах есть столбцы с такими названиями)
# # или left_on, right_on = 'название столбца в правом объекте'
# # но появится избыточный столбец, удалить можно drop()
# # слияние по меткам индекса: left_index=True и/или right_index=True
# # how - тип слияния ('inner' - пересечение, 'outer' - все вместе, 'left' - Nan только в правом столбце)
# # suffixes = ['_R', '_T'] - для конфликтующих столбцов
# # СЛИЯНИЕ С УЧЕТОМ ОДИНАКОВЫХ СТОЛБЦОВ
# # cols_to_use = df2.columns.difference(df1.columns)  # нахождение разных столбцов
# # dfNew = merge(df, df2[cols_to_use], left_index=True, right_index=True, how='outer')


# ПРЕОБРАЗОВАНИЕ ДАННЫХ СТОЛБЦОВ ТАБЛИЦЫ PIVOT()
# преобразуем столбцы для лучшего визуального восприятие
# new_table = table.pivot(index='col1интервал',
#                         axis='col2оси',
#                         values='col3данные')


# СТЫКОВКА И РАСТЫКОВКА
# !!! стыковка и растыковка ВСЕГДА помещает уровни в самые внутренние уровни другого индекса
# в состыкованных данных быстрее осуществляется поиск данных
# df.stack() - помещает столбцы в еще одни уровени индекса
# df.unstack(level=1) - пермещает самый внутренний уровень индекса строк в новый уровень индексов столбцов
# level=1 - для выбора уровня индекса расстыковки
# df.unstack(['name1', 'name2']) - расстыковка сразу несколько уровней индекса с указанием имен


# # ФУНКЦИЯ APPLY
# # print(df.apply(summ)) - суммирование по столбцам по умолчанию
# # print(df.apply(summ, axis=1)) - по строкам
# df.apply(lambda x: x.b + x.c, axis=1) - суммирование 2-х столбцов
# df.applymap(lambda: x: '%.2f' % x) - форматирование для каждого значения

# # ЗАДАНИЕ КЛЮЧА РАЗБИЕНИЯ
# # L = [4, 4, 4, 4, 5, 1] - через список
# # print(df.groupby(L).sum())
#
# df2 = df.set_index('key')  # задание индекса у массива, как буквы в столбце key
# mapping = {'A': 'blo', 'B': 'ibi', 'C': 'ibi'}  # через словарь
# # print(df2.groupby(mapping).sum())
#
# # передача в groupby любой функции
# # print(df2.groupby(str.lower).mean())
# # комбинирование
# # print(df2.groupby([str.lower, mapping]).mean())  # будет два столбца индексы: a,b,c и blo, ibi, ibi
#
#
# # СВОДНЫЕ ТАБЛИЦЫ
# # titanic.groupby('sex')[['survived']].mean() выжившие в зависимости от пола
# # titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
# # группировка по классу и полу, среднее знач выживших, unstack для раскрытия таблицы, т.к два столбца индкс
# # titanic.pivot_table('survived', index='sex', columns='class') или эквивалент
#
# # age = pd.cut(titanic['age'], [0, 18, 80])  # разбиение на интервалы
# # titanic.pivot_table('survived', ['sex', age], 'class')
#
# # DataFrame.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None,
# #                       margins=False, dropna=True, margins_name='All')
# # aggfunc - тип агрегирования (aggfunc={'survived':sum, 'fare':'mean'})
# # margins=True  # вычисление итогов по всем строкам и столбцам
#
#
# # ВЕКТОРИЗОВАННЫЕ ОПЕРАЦИИ НАД СТРОКАМИ
# # names.str.capitalize()
#
#
# # УВЕЛИЧЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ EVAL(), QUERY()
# # ФУНКЦИЯ EVAL()
# # pd.eval('d1 + d2 + d3') - быстрее на 50%
# # pd.eval('d[3] + d.iloc[19]')
# # result2 = pd.eval("(df.A + df.B) / (df.C - 1)") - можно ссылаться к столбцам напрямую по имени
# # df.eval('(A + B)/C') - или так
# # df.eval('D = (A + B) / C', inplace=True) - создание нового столбца
#
# # new_result = df.mean(1)
# # result2 = df.eval('A + @new_result')
#
# # ФУНЦКИЯ QUERY()
# # result2 = df.query('A < 0.5 and B < 0.5')
# # result2 = df.query('A < @Cmean and B < @Cmean')

# ШАГИ ПО ПОВЫШЕНИЮ ПРОИЗВОДИТЕЛЬНОСТИ
# I
# 1) самое медленное - это базовое итерирование 696мс
# 2) метод iterrows() в 3 раза быстрее for index, row in df.iterrows(): 215мс
# 3) лучше apply() засчет оптимизаций df.apply(lambda row: fun(row)) 81мс
# векторизация лучше скалярных операций которые использовались выше
# 4) векторизованная реализация выполнения функции, передавая объекты Series целиком fun(df['col'] ... df[]) 2мс
# 5) векторизация с помощью массивов NumPy (np использует оптимизированный, предварительно скомпилированный
# код на языке С (если не нужно ссылаться на значения по индексу, когда индекс не важен)
# необходимо преобразовать серии в массивы NumPy .values - fun(df['col'].values) - в 6 раз быстрее 0.37мс
# II
# после всех верхних улучшений (оптимизация исходного питоновского кода)
# необходимо использовать CPython в 100 раз быстрее 5)


# # DATETIME
# СОЗДАНИЕ
# datetime.datetime(2019, 4, 22) - 2019-04-22 00:00:00

# TIMESTAMP - падосовский временной объект, более точный
# СОЗДАНИЕ
# pd.Timestamp(2020, 5, 22) или pd.Timestamp('2020-5-22 17:30')
# pd.Timestamp('now') - сейчас время
# TIMEDELTA
# pd.Timedelta(days=1) - 1 days 00:00:00
# DATETIMEINDEX - если индекс пандаса даты
# a = pd.Series(2, [datetime.datetime(2019, 2, 3)])
# print(type(a.index)) - весь индекс <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
# print(type(a.index[0])) - отдельный элемент индекса <class 'pandas._libs.tslibs.timestamps.Timestamp'>
# преобразование полследовательности объектов в datetimeindex
# # pd.to_datetime(['2019-2-3',
#                  '2014.4.5',
#                  None], errors='coerce') - если не распознает то Not a time -  NaT
# если не распознает и чтобы не выдавало исключение, а было NaT, то errors='coerce'
# ОБРАЩЕНИЕ
# df['2019-01-01'] - вывод одной строки
# df['2019-01-01':'2019-02-01'] - срез
# более конкретно
# df['2019'] - определенного года также и для месяца, дня и т.д. (df['2019-01']
# СОЗДАНИЕ ДИАПАЗОНА
# dates = pd.date_range('2019-01-01', '2019-01-07') # DatetimeIndex
# pd.date_range('2018-01-01', periods=5, freq='H') - 5 интервалов времени через час
# D - день, Т - минута, все остальные частоты стр.274
# СМЕЩЕНИЯ класс DateOffset
# создание
# sm = pd.DateOffset(day=1) - создание смещения в 1 день (datetime(2018, 8, 3) + sm)=на 1 день больше
# date + класс смещения = новая дата (для каждой частоты свой подкласс стр.276)
# dates.freq - вывод частоты временных меток в индексе (для каждой частоты свой подкласс стр.276)
# ПЕРИОДЫ PERIOD
# per = pd.Period('2019-08-22', freq='M') - точка отсчета и период в 1 месяц
# per + 1 - смещение периода на 1 единице периода, т.е. будет слудеющий месяц
# PERIODINDEX
# pd.period_range(1/1/2013, 31/12/2013, freq='M') - тот же самый Datetimeidnex только объект PeriodIndex

# # перевод в строку
# # date.strftime('%d-%m-%Y %H:%M') %d - день, %m - мес, %Y - год, %H - час (24), %M - мин, %S - сек,
# # date.strftime('%x')  %с - время и дата, %x - дата, %X - время
# # date.strftime('%A') - %A - полное название дня недели, %a - сокр, %s - в виде числа, МЕСЯЦЫ: %B, %b, %m
# # date_string = '21 September. 1999'
# # dateobject = datetime.strptime(date_string, '%d %B. %Y')
# # duration_time = datetime.timedelta(days=12, seconds = 33)


# ЧАСОВЫЕ ПОЯСА
# по умолчанию часовые пояса не учитываются
import pytz
# from pytz import common_timezones - импорт всех часовых поясов
# print(common_timezones) - вывод всех часовых поясов
# pd.Timestamp('2018-08-01 18:20:00', tz='US/Mountain') - 2018-08-01 18:20:00-06:00
# dates = pd.date_range('2019-01-01', '2019-01-07', tz='US/Mountain') # DatetimeIndex
# ПРЕОБРАЗОВАНИЕ ЧАСОВОГО ПОЯСА
# table.tz_convert('US/Mountain')

# ОПЕРАЦИИ С ВРЕМЕННЫМИ РЯДАМИ
# ПЕРЕМЕЩЕНИЕ ДАННЫХ ПО ВРЕМЕННОЙ ОСИ
# table.shift(1) - перемещение ЗНАЧЕНИЙ на один день ввперед (назад (-1))
# table / table.shift(1) - 1 - процентное изменение
# table.shift(1, freq='B') - свиг временной оси на 1 рабочий день вперед
# ПРЕОБРАЗОВАНИЕ ЧАСТОТЫ ВРЕМЕННОГО РЯДА
# индекс будет начинаться с метки исходной серии с указанной изменной частотой до конечной метки исходной серии.
# из-за выравнивания в случае увеличения шага частоты некоторые данные удалются, а если шаг уменьшить (с часа до
# минуты), то меток больше, а данных остается столько же, поэтому по некоторым позициям Nan
# table.asfreq('новая частота', fill_value=9) - новые Nan заполнятся 9
# table.asfreq('новая частота', method='ffill'/'bfill') - для заполнения отсутсвующих элементов
# ИЗМЕНЕНИЕ ШАГА ДИСКРЕДИТАЦИИ
# понижающая передискредитация - уменьшение частоты, более низкая частота, итоговое меньшее количество элементов
# передискретизация делит на интервалы, и для данных каждого интервала применяет операцию
# table.resample('1Min', close='left').mean() - было 1сек частота, будет 1мин
# ('2019-05-01', 2019-06-14'] - интервал закрытый справа - close='right' (по умол. левый)
# table.resample('1Min').first() - вывод первого значения интервала
# для повышающей передискредитации отсутствующие значения будут Nan
# table.resample('S').bfill() - обратное заполнение пропущенных значений (или interpolate())
# table.resample('H').ohlc() - open high low close - для понижающей дискредитации
# группировка по столбцу и передискредитация по индексу (заполнение пропусков 0)
# df.groupby('a').resample('M').first().fillna(0)


# ЗАГРУЗКА ДАННЫХ ИЗВНЕ
# CSV
# CSV - файл данных с разделителями (по умолчанию запятая).
# df = pd.read_csv('название файла, или путь', sep=';', index_col='название столбца индекса',
#                  parse_dates=['date'], nrows=1)  # загрузка количества строк начинаю с первой (только первую)
# df = pd.read_csv('C:/data/file.csv', usecols=[0, 2, 6]) - считывание только тех столбцов у которых инексы 0,2,6
# df = pd.read_csv('file.csv', usecols=['aa', 'bb']) - считывание только указанных столбцов
# df = pd.read_csv('file.csv', skiprows=[0, 1]) - пропуск указанных строк
# df = pd.read_csv('file.csv', skipfooter=2, engine='python') - пропуск 2-х последних строк, без engine не работет
# ИЗМЕНЕНИЕ ТИПА СТОЛБЦА
# df = pd.read_csv('file.csv', dtype={'col': np.float64})
# НАЗВАНИЕ СТОЛБЦОВ
# df = pd.read_csv('file.csv', names=['asf', 'asdf'], header=0) - header какая строка-заголовок
# СОХРАНЕНИЕ
# df.to_csv('Название нового файла', index_label='название столбца с индексами', sep=';')  # сохранение
# portfel.to_csv('tr.csv', index=False, mode='a', header=False) пересохранение с добавлением,
# без учета индекса пандаса, вместо mode='w' будет 'a', и не учитываем строки заголовка пандасa

# XLSX - все параметры от csv также работают
# df = pd.read_excel('file.xlsx', sheet_name='название страницы в excel')  # можно загрузить только одну страницу
# df.to_excel('название.xlsx', index_label='sadf')  # сохранение
# ЗАПИСЬ НЕСКОЛЬКИХ ЛИСТОВ
# from pandas import ExcelWriter
# with ExcelWriter('file.xlsx') as writer:
#     df1.to_excel(writer, sheet_name='one')
#     df2.to_excel(writer, sheet_name='two')

# JSON
# import json
# ОДИНОЧНЫЕ ' НЕ ЯВЛЯЮТСЯ JSON, НУЖНО ДВОЙНЫЕ
# str = str.replace("\'", "\"")

# СОХРАНЕНИЕ
# json.dump(object, fp=open('textfile.txt', 'w'), indent=4) - сохранение json в файл (object - то что хотим сохранить
# fp - туда куда хотим, т.е. файл на компе, indent=4 нужне для красивого форматирования (лучше читается глазами)
# после надо закрыть файл fp, чтобы можно было дальше его использовать
# поэтому лучше так
# with open('textfile.json', 'w') as file:
#     json.dump(object, file, indent=4)
# ОТКРЫТИЕ
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

# ЗАГРУЗКА В PANDAS
# pd.read_json('C:/folder/file.json')
# df = pd.read_json(response2)
# df = pd.read_json('https://www.metaweather.com/api/location/2122265/2020/06/20/') # можно и так


# ежедневное процентное изменение
# table_pc = (table / table.shift(1)) - 1
# ежедневная накопленная доходность
# (1+table_pc).cumprod()


# BIG DATA
# df.info(memory_usage='deep') - вывод точной информации о потреблении памяти
# внутре пандас группирует столбцы в блоки значений одного и того же типа
# столбцы с числами объединяются как одни многомерный массив NumPy
# тип object(строки, и смешанный тип) больше всех памяти расходует
# uint8=0-255, int8=-128_127, int16=-32768_32767 (числа без знака хранятся более эффективно)
# ПОНИЖАЮЩЕЕ ПРЕОБРАЗОВАНИЕ ДЛЯ INT
# df_int = df1.select_dtypes(include=['int']) - выбираем столбцы типа int (у меня почему-то не работает)
# convert = df_int.apply(pd.to_numeric, downcast='unsigned') - применяем функцию понижения
# ТОЖЕ САМОЕ ДЛЯ FLOAT
# df_int = df1.select_dtypes(include=['float']) - тут уже работает
# convert = df_int.apply(pd.to_numeric, downcast='float') - float64 -> float32
# СПОСОБЫ ХРАНЕНИЯ ЧИСЛОВЫХ И СТРОКОВЫХ ЗНАЧЕНИЙ
# строки в серии занимают столько же места сколько занимали бы отдельно в Python
# изменение типа столбца object на category - выявляет уникальные значения и присваивает им целое число
# df['col'].astype('category') - на 98% может снизить потребление памяти
# !!! использовать только если количество уникальных значений не более 50% от всех значений в столбце
# !!! также нельзя применять функции такие, как max(), min() и т.д.
# ИЗМЕНЕНИЕ ТИПА СТОЛБЦА ПРИ ЗАГРУЗКЕ
# pd.read_csv('file.csv', dtype=dict, parse_dates=['date']) - где dict - словарь ключи это название столбцов,
# а значения нужные типы

# Функция для определения потребляемой памяти
# def mem_useg(pandas_obj):
#     if isinstance(pandas_obj, pd.DataFrame):
#         usage_b = pandas_obj.memory_usage(deep=True).sum()
#     else:
#         usage_b = pandas_obj.memory_usage(deep=True)
#     usage_mb = usage_b / 1024 ** 2
#     return "{:03.19f} MB".format(usage_mb)
# print(mem_useg(df1['c']))

# краткий план
# 1) удаление ненужных столбцов
# 2) изменение типов столбцов
# 3) обработка редких категорий (вычисляем количество категорий) самые редкие категории объединяем в одну общую
#    категорию OTHER - table.at[table['col'] == 'редкая категория', 'col'] == 'OTHER' к остальным также
# 4) при прогнозном моделировании дальше необходимо разбить данные на обучающую и контрольную выборки
# 5) ампутация пропусков отдельно у каждой выборки (пропуски заполняются медианой и т.п. вычисленные на
#    ОБУЧАЮЩЕЙ ВЫБОРКЕ. Категории заменяются на моду категорий
#    df['col'] = df['col'].fillna('название категории')