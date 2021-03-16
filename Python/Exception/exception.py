# ИСКЛЮЧЕНИЯ (EXCEPTION)
# Исключения всегда являются экземплярами класса
# ВОЗВРАТИТЬСЯ В КОД, КОТОРЫЙ СГЕНЕРИРОВАЛ ОШИБКУ, ВОЗМОЖНОСТИ НЕТ


# КОНСТРУКЦИЯ
# try:
#     print(a, 'выполнение кода')
#     a = 4  - объекты, определенны в конструкции try не видны в других областях
# except NameError:
#     print('обработка исключения NameError')
# except (TypeError, SyntaxError):
#     print('обработка исключений')
# except Exception:
#     print('обработка всех исключений')
# except:
#     print('обработка всех исключений')
# else:
#     print('выполнится только, если исключений не возникало')
# finally:
#     print('выполнится в любом случае')
# print('продолжение выполнения кода после исключения')


# ОПЕРАТОР RAISE
# raise класс  -  генерирует экз класса
# raise экземпляр  -  генерирует и создает экз класса: создает экз
# raise  -  повторно генерирует самое последнее исключение

# генерация исключения
# raise IndexError - класс (экз создается неявно)
# raise IndexError() - экз (создается явно в операторе)

# созданние экза заранее
# exc = IndexError()
# raise exc
# или
# exc = [IndexError(), TypeError()]
# raise exc[0]


# ОБЛАСТИ ВИДИМОСТИ И ПЕРЕМЕННЫЕ
# except IndexError as exc  -  exc присваивается сгенерированный экземпляр
# x = 1
# try:
#     1 / 0
# except Exception as x:
#     print(x, x.args)
#     y = x
# print(y)  # division by zero
# print(x)  # NameError: name 'x' is not defined -- выдаст ошибку, так как х удален


# СЦЕПЛЕНИЕ ИСКЛЮЧЕНИЙ: RAISE FROM
# если старое исключение не перехвачено, то выведутся оба исключения
# try:
#     1 / 0
# except Exception as E:
#     raise TypeError from E  # явно сцепленные исключения

# неявно сцепленные исключения
# try:
#     1 / 0
# except:
#     badname

# ОПЕРАТОР ASSERT
# в основном применяется для улавливания нарушений ограничений, определяемых пользователем
# assert test, data
# аналог
# if __debug__:
#     if not test:
#         raise AssertionError(data)

# def f(x):
#     assert x < 0, 'ошибка, х должна быть меньше 0'
#     return x ** 2
# f(1)  # AssertionError: ошибка, х должна быть меньше 0


# ДИСПЕТЧЕРЫ КОНТЕКСТОВ: WHILE/AS
# with open(r'C:\music\data') as myfile:
#     for line in myfile:
#         print(line)

# ПРОТОКОЛ УПРАВЛЕНИЯ КОНТЕКСТАМИ

# МНОЖЕСТВО ДИСПЕТЧЕРОВ КОНТЕКСТОВ
# with open('data') as fin, open('res', 'w') as fout:
#     for line in fin:
#         if 'some key' in line:
#             fout.write(line)
#     for pair in zip(fin, fout):
#         print(pair)

# with A() as a, B() as b:
#     ...операторы...
#     аналогичная запись
# with A() as a:
#     with B() as b:
#         ...операторы...

# class InOutBlock:
#
#     def __enter__(self):
#         print('Входим в блок кода')
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         print(f'Выходим из блока кода {exc_type}, {exc_val}, {exc_tb}')
#         return True  # возвращаем истину, что бы погасить полет исключения
#
#
# with InOutBlock() as in_out:
#     print('Работаем...')
#     a = bla_bla / number
#     print('Вычислили значение')
# print('После контекстного менеджера')




# ИЕРАРХИИ ИСКЛЮЧЕНИЙ
# class General(Exception): pass
# class Specific1(General): pass
# class Specific2(General): pass
# import sys
# print(sys.exc_info()[0])  # инфо об классе исключения
# print(x.__class__)  # если в исключениях внутри уже указан класс


# СПЕЦИАЛЬНОЕ ОТОБРАЖЕНИЕ ПРИ ВЫВОДЕ
# class MyBad(Exception):
#     def __str__(self):
#         return 'Всегда смотри на светлую сторону жизни!'
#
# try:
#     raise MyBad()
# except MyBad as X:
#     print(X)



# class FormatError(Exception):
#     def __init__(self, line, file):
#         self.line = line
#         self.file = file
#
# def parser():
#     raise FormatError(42, file='text.txt')
#
# try:
#     parser()
# except FormatError as X:
#     print('Error at: %s %s' % (X.file, X.line))  # Error at: text.txt 42

# ИЛИ

# class FormatError(Exception): pass
#
# def parser():
#     raise FormatError(42, 'text.txt')
#
# try:
#     parser()
# except FormatError as x:
#     print('Error at: %s %s %s' % (x.args, x.args[0], x.args[1]))  # Error at: (42, 'text.txt') 42 text.txt


