# СТРОКИ - ПОЗИЦИОННЫЕ УПОРЯДОЧЕННЫЕ КОЛЛЕКЦИИ (НЕИЗМЕНЯЕМЫЕ ОБЪЕКТЫ)

# s = 'shrubbery'
# L = list(s)
# print(L)
# ['s', 'h', 'r', 'u', 'b', 'b', 'e', 'r', 'y']

# s.replace('pa', 'XYZ')

# разбить по разделителю с спиок подстрок
# line = 'aaa,bbb,ccc,ddd'
# line.split(',') ---- ['aaa','bbb', ....]

# s.upper() - преобразовать в верхний регистр
# s.lower() - преобразование в нижний регистр

# s.isalpha() - Вернёт True, если в строке хотя бы один символ и все символы строки являются буквами, иначе — False.
# s.isdigit() - Вернёт True, если в строке хотя бы один символ и все символы строки являются цифрами, иначе — False.

# 'abca'.rstrip('ac') --- 'ab' - Возвращает копию указанной строки, с конца которой устранены указанные символы.
# 'abca'.strip('ac')  # 'b'
# 'abca'.ltrip('ab')  # 'ac'

# help(str)

d = 'asdfdfdsd\nssfsaf\tdfdsf'
# \n с новой строки, \t пробел в тексте

a = r'C:\text\new'
# вывод C:\text\new

# help(list)

# print(list(ord(x) for x in 'sddf')) -- номер символа в кодировке
# print(set(ord(x) for x in 'sddf'))

s = 'sdf dsfsdfd sfasdf'
s1 = 's,llll,ddd'
l = ['a', 'b', 'c']
# print(s)
# print(s.find('df', 5))

# print(s.rstrip(' '))
# print(s.replace('df', 'XY')) -- sXY dsfsXYd sfasXY

# print(list(s)) ['s', 'd', 'f', ' ', 'd', 's', 'f', 's', 'd', 'f', 'd', ' ', 's', 'f', 'a', 's', 'd', 'f']

# print(''.join(l)) -- abc

# print('РАЗДЕЛИТЕЛЬ'.join(l)) -- aРАЗДЕЛИТЕЛЬbРАЗДЕЛИТЕЛЬc

# print(s1.split(',')) -- ['s', 'llll', 'ddd'] по умолчанию пробел

# print('df %2d sdf, %s, sdfdfd %s' % (1, 'dsfd', [1, 2, 3]))  # -- выражение форматирования

# y = '{xax}, {} and {}' -- вызов метода форматирования -- метод фораматирования
# x = y.format('dsfd', 'dfddsss', xax=[1, 'sdsd'])
# print(x)

import sys
# a = 'my \'{[key1]:>15}\' runs {sys.platform} dfdf {ma[key2]}'.format({'key1':'325'}, sys=sys, ma={'key2':'34343'})
# my '         325' runs win32 dfdf 34343

# print('\'{ma[key]:<012.3E}\' -- '.format(ma={'key':3.1443434})) -- '3.144E+00000' --
# print('\'{ma[key]:>012.{}f}\' -- '.format(3, ma={'key':3.1443434})) -- '00000003.144' --

# print(format(3.145456, '>015.3f')) -- 00000000003.145

# x = {'key1': 3489989898.343434344, 'key2': 45}
# print('{key1:<04,.3f} dsfsd {key2:>04}'.format(**x)) -- 3,489,989,898.343 dsfsd 0045

# print(str.format('{}, {}', 88, 89)) -- использование функции

# ФУНКЦИЯ EVAL
# x=1
# print(eval('x+1')) -- 2

# a = '2,3,4'
# c = '{\'n\': 3, \'m\': 4}'
# print(eval(a)) -- (2, 3, 4)
# print(list(eval(a))) -- [2, 3, 4]
# print(eval(c)) -- {'n': 3, 'm': 4}
# print(eval(input()))

# УДАЛЕНИЕ ЗНАКОВ ПРЕПИНАНИЯ
import string
# a = 'фвыаб, выа , вавыа !!!
# print(a)  # 'фвыаб, выа , вавыа !!!'
# print(''.join([i for i in a if i not in string.punctuation]))  # 'фвыаб выа  вавыа '





# РЕГУРЛЯРНЫЕ ВЫРАЖЕНИЯ
import re
# test_string = 'какой-то текст или.... с любыми символами'
# reg_expr = r'\w+'  # регулярное выражение
# reg_expr_compiled = re.compile(reg_expr)  # компилируем регулярное выражение
# res = reg_expr_compiled.findall(test_string)  # поиск всех совпадений

# ИЛИ
# res = re.findall('регулярное выражение', 'текст в котором ищем')

# ПРИМЕРЫ СПЕЦСИМВОЛОВ
# . Любой символ
# \w Любая буква (то, что может быть частью слова), а также цифры и _
# \W Всё, что не входит в \w
# \d Любая цифра
# \D Всё, что не входит в \d
# \b граница слова
# \s любой пробельный символ
# \S любой непробельный символ
# […] можно указывать любой набор символов (для поиска только одного символа) [0123456] или [0-6], [-=?a-zA-Z0-9]

# Кроме спецсимволов можно использовать т.н. квантификаторы - указатели количества
#
# ? - от 0 до 1  {0,1}
# + - одно или более вхождений {1,}
# * ноль или больше вхождений  {0,}
# {m,n} сколько раз повторяется (жадный и мажорны) т.е. наибольшую последовательность ищет, наоборот {m,n}?
# {m} ровно n раз
# {m,} m и более    - для них тоже есть минорный режим }?
# {,n} не более n
# ^ начало вхождения
# $ конец вхождения
# () - группирующие скобки. Позволяет искать подстроки

# просто указывать спецсимволы для поиска нельзя, их не найдет, нужно перед ним поставить "\"
# re.find_all(r'\{asdf', stroka)

# [eЕ]д[ау] - еда, Еда, еду, Еду
# [-0-9][0-9] - -5, 69
# g{2, 5} - gg, ggg, ggggg
# g{2, 5}? - gg, gg, gg, gg
# стеклянн?ый  -  стеклянный, стекляный (н? - одна н или ноль)
# \w+\s*=\s*[^;]+ - все буквы одна и больше, пробел от нуля и больше, равно, пробел от нуля и больше, все символы
# кроме ; от одного и больше
# "<img src='bg.jpg'> в тексте</p>" - findall(r'<img.*>', ) вернет все выражение <img src='bg.jpg'> в тексте</p>
# но если исп минор ? findall(r'<img.*?>', ), то <img src='bg.jpg'>

# [^>]*?src - все символы (кроме >) дальше от 0 до бесконечности, пока не встрети src, и обязательно в миноре, должен
# минимальную последовательность

# СОХРАНЯЮЩИЕ СКОБКИ () -- НЕСОХРАНЯЮЩИЕ (?:)
# text = 'lat = 5, lon=7'
# findall(r'lat\s*=\s*\d+|lon\s*=\s*\d+', text) -- ['lat = 5', 'lon=7']
# но лучше так (r'(?:lat|lon)\s*=\s*\d+', text) -- ['lat = 5', 'lon=7']
# (r'(lat|lon)\s*=\s*\d+', text) -- ['lat', 'lon']
# (r'(lat|lon)\s*=\s*(\d+)', text) -- [('lat', '5'), ('lon', '7')]

# ([\"'])(.+?)\1 -- \1 - повторение того что указано в 1 сохраняющих скобках
# аналогия но с именем сохраняющей скобки (?P<q>[\"'])(.+?)(?P=q)

# \b(слово1|word2|word3)\b  --поиск слова, одного из 3, \b применяется к каждому слову


# SEARCH - ищет только первое попавшееся вхождение, другие игнорирует
text = '<font color=#CC0000 bg=#ffffff>'
match = re.search(r'(\w+)=(#[\da-fA-F]{6}\b)', text)  # получаем объекть match
# print(match)  # <re.Match object; span=(6, 19), match='color=#CC0000'>
# print(match.group(0, 1, 2))  # ('color=#CC0000', 'color', '#CC0000')
# print(match.group())  # color=#CC0000
# print(match.lastindex)  # 2
match = re.search(r'(?P<key>\w+)=(?P<value>#[\da-fA-F]{6})\b', text)
# print(match.groupdict())  # {'key': 'color', 'value': '#CC0000'}
# print(match.expand(r'\g<key>:\g<value>'))  # color:#CC0000
# print(match.expand(r'\1:\2'))  # color:#CC0000

# чтобы найти другие вхождения используем FINDITER
# for match in re.finditer(r'(?P<key>\w+)=(?P<value>#[\da-fA-F]{6})\b', text):
#     print(match)



# ВЕКТОРИЗАЦИЯ ТЕКСТА
corpus = []
# regular_expr = r'\w+'  # разбивка текста на слова
# text_by_words = re.findall(r'\w+', text)

import pymorphy2 as pm  # библиотека для перевода слова в нормальную форму

morth = pm.MorphAnalyzer()  # создание объекта для перевода слова в норм форму
# word = 'Теперь'
# parsed_token = morth.parse(word)  # перевод слова
# print(parsed_token)  # [Parse(word='друзьями', tag=OpencorporaTag('NOUN,anim,masc plur,ablt'),
# # normal_form='друг', score=1.0, methods_stack=((DictionaryAnalyzer(), 'друзьями', 1423, 10),))]
# print(parsed_token[0])  # для извлечения из списка (результат без [])
# normal_form = parsed_token[0].tag  # вывод из Parse normal_form='друг'
# print(normal_form)  # друг

# print('Name' in parsed_token[0].tag)

# НАХОЖДЕНИЕ ИМЕНИ И ФАМИЛИИ:
# регулярка - её нужно поправить
# reg_expr = r'([А-ЯЁ][а-яё]+\s[А-ЯЁ][а-яё]+)'
# # компилируем регулярное выражение
# reg_expr_compiled = re.compile(reg_expr)
# morph = pm.MorphAnalyzer()
# for g in reg_expr_compiled.findall(text):
#     a = 0  # устанавливаем счетчик
#     for y in re.findall(r'\w+', g):  # разделяем имя и фамилию
#         x = morph.parse(y)
#         if 'Name' in x[0].tag or 'masc' in x[0].tag: # производим проверку
#             a += 1  # слово верно, оставляем
#         else: # иначе берем другую пару
#             break
#     if a == 2: # при правильности обоих слов, выводим имя и фамилию
#         print(g)


genre_dict = {
    'комедия': ['сатирический', 'авантюрный', 'забавный'],
    'мелодрама': ['выбор', 'позор'],
    'сказка': ['приключения', 'милый', 'семейный'],
    'детектив': ['тайна', 'разгадать', 'загадочный'],
    'триллер': ['ужас', 'зловещий', 'нерв']
}
import itertools
nested_genres =[[(i, j) for j in genre_dict[i]] for i in genre_dict]
print(nested_genres)
flatten_genres = list(itertools.chain(*nested_genres))
print(flatten_genres)


