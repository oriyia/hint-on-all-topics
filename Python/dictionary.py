# СЛОВАРИ - НЕУПОРЯДОЧЕННЫЕ КОЛЛЕКЦИИ ОБЪЕКТОВ

# print(dict(name='bob', age=23, job='dev')) -- {'name': 'bob', 'age': 23, 'job': 'dev'}

# print(dict(zip(['name','age','job'],['bob',24,'dev']))) -- {'name': 'bob', 'age': 24, 'job': 'dev'}
# print({k: v for (k, v) in zip(['a', 'b'], [1, 2])})

# print(dict.fromkeys(['a', 'b'], 0)) -- {'a': 0, 'b': 0}

d = {'2015_4': 1444, '2015_2': 1500, '2015_1': 800, '2015_3': 600}
d1 = {'2015_4': 1444, '2015_6': 1500, '2015_1': 800, '2015_5': 600}
# print(d.get('c',4)) -- 4 если ключ не найден в словаре

# print(list(d.items())[0]) -- получение первой пары ('x', 1)

# print(list(d.keys())) -- получение ключей ['x', 'y']

# print(d['2015_2'])

# print(sorted(d.items()))
# print(sorted(list(d.keys())))

# print(list(d.keys() & d1.keys())) -- общие ключи

# print(list(key for key in d)) -- вывод ключей

# print(list(title for (key, title) in d.items())) -- вывод содержимого

# try:
#     print(d[2018])        -- обработка исключения
# except KeyError:
#     print('нихрена подобного')

# for key in D:
for x in range(4):
    print(x)
    dict

