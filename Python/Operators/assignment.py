# ОПЕРАТОР ПРИСВАИВАНИЯ
string = 'spam'
[a, b] = ['a', 'b']
# a, b, c, d = 'spam' -- s p a m
# a, *b = 'spam' -- s ['p', 'a', 'm']
# (a, b), c = string[:2], string[2:]
# print(a, b , c) -- s p am

# a, *b, c = string
# print(a,b,c) -- s ['p', 'a'] m

# ДОПОЛНЕННОЕ ПРИСВАИВАНИЕ
L = [1, 2, 3]
# L += [4, 5] -- [1, 2, 3, 4, 5] то же самое что и extend

# print(L.append(4)) -- результат будет None, метод возвращает не измененный список

a = ('asdfdfdsfdfdsfds',
     'asdfasdfdfsdafdsf',
     'adsfdfdfasdf')
x = 1; y = 2; z = 3

# print([object,][, sep=''][, end='\n'][, file=sys.stdout][, flush=False])
# print(x, y, z, sep='...', file=open('data.txt', 'w')) -- вывод в файл
# print(open('data.txt').read()) -- отображение текста из файла
# или
# import sys
# sys.stdout = open('data.txt', 'a')
# print(x, y, z) -- запись в файл
# или
# log = open('data.txt', 'a')
# print(x, y, z, file=log) -- вывод в файл

