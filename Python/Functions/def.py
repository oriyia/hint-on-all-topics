# def структурирование программ, и их много кратный вызов
# доведение до максимума многократного использования кода
# сведение к минимуму избыточности
# def создает объект и присваивает его имени
# ПОЛИМОРФИЗМ - смысл операции зависит от обрабатываемых ею объектов

# меньше использовать глобальных переменных
# меньше использовать неявные межфайловые зависимости

# локальная область видимости - локальная область видимости объемлющих функций - глобальная - встроенная (builtins)

# ФАБРИЧНЫЕ ФУНЦИИ - ЗАМЫКАНИЯ (сохранение состояния)
def maker(N):
    def action(X):
        return X ** N

    return action


f = maker(2)  # <function maker.<locals>.action at 0x00DEEC88> возврат action без выполнения
# print(f(3)) -- 9 = 2 * 3
g = maker(3)
# print(g(4)) -- 64 = 4 * 3

def maker(N):
    return lambda X: X ** N
h = maker(4)
# print(h(4)) -- 256

# Лучше так
# def f1():
#     x = 88
#     f2()
# def f2(x):
#     print(x)

def func():  # lambda выражение вкладывает в объемлющий оператор def
    x = 4
    return lambda n: x ** n
x = func()
# print(x(2))

def makeac():
    acts = []
    for i in range(5):
        acts.append(lambda x, i=i: i ** x)  # передача текущего значения через стандартное значения
    return acts

acts = makeac()
# print(acts[0](2))


# NONLOCAL
# имена, перечисленные в операторе nonlocal, уже должны быть определены в объемлющем операторе def
# nonlocal означает "полностью пропустить мою локальную область"

def tester(start):
    state = start
    def nested(label):
        nonlocal state
        print(label, state)
        state += 1
    return nested

f = tester(3)
# print(f('spam'))

# def asd1():
#     x=1
#     print(x)  # x = 1
#     y = 4
#     def add():
#         x=2
#         def aaa():
#             nonlocal x
#             nonlocal y
#             print(x)  # x = 2
#             print(y)  # y = 4
#             x = 3
#             print(x)  # x = 3
#         aaa()
#     add()
# asd1()


# АРГУМЕНТЫ
# избегать модификации изменяемых аргументов

def fun(x, y):
    x = 1
    y = [2, 3]
    return x, y

x = 3
y = [4, 6]
x, y = fun(x, y)
# print(x, y) -- 1 [2, 3]

# def func(a, *b, c=6, **d):
# func(1, *(2, 3), c=8, **dict(x=5, y=6))


# ОБЪЕКТ ФУНКЦИЙ: АТРИБУТЫ И АННОТАЦИИ
# передача функции в функцию
def func(message):
    print(message)

def newf(f, arg):
    f(arg)
# print(newf(func, 'sdfasdf'))

# ИЛИ списки с циклом for
schedule = [(func, 'spam'), (func, 'asaaaaaa')]
for fun, arg in schedule:
    fun(arg)

# АТРИБУТЫ ФУНКЦИЙ
# print(dir(func)) -- ['__annotations__', '__call__', '__class__', '__closure__', '__code__',
# print(len(dir(func))) -- количество атрибутов 35

# LAMBDA - создает функцию, которая вызвается позже, но возвращает сам объект функции (АНОНИМНЫЕ)
# это не оператор, а выражение, поэтому может находиться в списках или в аргументах вызова функции
f = lambda a, b, c: a + b +c
# print(f(1, 1, 1)) -- 3

x = (lambda a='fa', b='th', c='er': a + b + c)
# print(x()) -- father

def maker(N):
    return lambda X: X ** N

h = maker(4)
# print(h(4)) -- 256

# ТАБЛИЦЫ ДЕЙСТВИЙ
L = [lambda x: x ** 2,
     lambda x: x ** 3,
     lambda x: x ** 4]

for f in L:
    pass # чтобы ошибки не было
#     print(f(2)) # -- 4, 8, 16
# print(L[0](2)) -- 4

key = 'got'
x = 2
a = {'hot': (lambda x: x + 2),
     'got': (lambda x: x * 8),
     'top': (lambda x: x ** 6)}[key](x)
# print(a) -- 16

lower = (lambda x, y: x if x < y else y)
# print(lower('aa', 'bb'))

import sys
showall = lambda x: list(map(sys.stdout.write, x))
showall = lambda x: [sys.stdout.write(line) for line in x]
showall = lambda x: [print(line, end='') for line in x]
showall = lambda x: print(*x, sep='', end='')
t = showall(('afadf\n', 'sdfa\n'))  # afadf sdfa

# ВЛОЖЕННЫХ LAMBDA ЛУЧШЕ ИЗБЕГАТЬ, ПЛОХАЯ ЧИТАБЕЛЬНОСТЬ
action = (lambda x: (lambda y: x + y))
act = action(99)
# print(act(1))
# print(((lambda x: (lambda y: x + y))(99))(1))

# MAP
# print(list(map((lambda x: x + 10), [1, 2, 3])))
# print(list(map(pow, [1, 2, 3], [4, 5, 6])))  # [1, 32, 729] возводит в степепь

# FILTER
# print(list(filter(lambda x: x > 0, range(-5, 5))))  # [1, 2, 3, 4]

# REDUCE
from functools import reduce
# print(reduce(lambda x, y: x + y, [1, 2, 3, 4]))  # 10
# print(reduce(lambda x, y: x * y, [1, 2, 3, 4]))  # 24

# ГЕНЕРАТОРНЫЕ ФУНКЦИИ
# сохранение состояния
# возвращают не значение, а новый объект генератора
# позволяют функциям избежать выполнения всей работы заранее

def sss(n):
    for i in range(n):
        yield i ** 4
    return
a = sss(4)
print(list(a))  # [0, 1, 16, 81]
print(list(a))  # []
print(tuple(a), tuple(a))  # (0, 1, 16, 81) () один раз можно вывести
print(list(sss(4)))  # [0, 1, 16, 81]
# for i in sss(4):
#     print(i, end=' : ')

def ups(line):
    for i in line.split(','):
        yield i.upper()

# print({j: n for (j, n) in enumerate(ups('aa,bb,cc'))})  # {0: 'AA', 1: 'BB', 2: 'CC'}

# ГЕНЕРАТОРНЫЕ ВЫРАЖЕНИЯ
# выражения включения, возвращают объект генератора
# (x ** 2 for x in range(4)) -- выводит итерируемый объект


