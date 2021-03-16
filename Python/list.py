# СПИСКИ - УПОРЯДОЧЕННЫЕ КОЛЛЕКЦИИ ПРОИЗВОЛЬНЫХ ОБЪЕКТОВ (МАССИВ ССЫЛОК НА ОБЪЕКТЫ)

# letters[2:5] = [] удаление значений из списка

# a = ['Mary', 'had', 'a', 'little', 'lamb']
# for i in range(len(a)):
#     print(i, a[i])

# list(range(4)) - для отображения элементов необходимо окружающий вызов list


# L = list('sdfdsf') -- ['s', 'd', 'f', 'd', 's', 'f']

# L = list(range(-3, 5)) -- [-3, -2, -1, 0, 1, 2, 3, 4]

# l = list(map(abs, [-1, -2, 0, 1, 2])) -- [1, 2, 0, 1, 2]

# L[1:1] = [6, 7] -- [1, 6, 7, 3, 3, 3] вставка, ничего не удаляет

# s[:0] = [10, 20] -- вставка в начало ничего не удаляя (без [])
# s[-1:] = [10, 20] -- вставка в конец ничего не удаляя (без [])

# L.extend([1, 2, 3]) -- вставка в конец списка

# l.sort(key=str.lower, reverse=True) -- метод
# sorted(L, key=str.lower, reverse=True) -- функция

# l.reverse()
# list(reversed(l)) -- функция инверсии

# lines = [line.rstrip() for line in open('file.py') if line[0] == 'p'] -- вывод всех строк начинающиеся с "р"

# [x + y for x in 'abc' for y in 'lmn']

# СПИСКОВЫЕ ВКЛЮЧЕНИЯ И МАТРИЦЫ
m = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
n = [[2, 2, 2],
     [3, 3, 3],
     [4, 4, 4]]

# print([m[i][i] for i in range(len(m))])  # [1, 5, 9] - диагональ
# print([m[i][len(m)-1-i] for i in range(len(m))])  # [3, 5, 7]

# print([col + 10 for row in m for col in row])  # [11, 12, 13, 14, 15, 16, 17, 18, 19]
# print([[col + 10 for col in row] for row in m])  # [[11, 12, 13], [14, 15, 16], [17, 18, 19]]

# print([[m[row][col] * n[row][col] for col in range(3)] for row in range(3)])  # [[2, 4, 6], [12, 15, 18], [28, 32...
# print([[col1 * col2 for (col1, col2) in zip(row1, row2)] for (row1, row2) in zip(m, n)])  # [[2, 4, 6], [12, 15...


# ФУНКЦИЯ SLICE
data = [1, 2, 3, 4, 5]
# print(data[slice(0, None, None)])  # [1, 2, 3, 4, 5]

# подсчет количества элементов в списке
# from collections import Counter
# x = 'male, male, female, female, male, male, female'
# y = x.split(', ')
# y = Counter(y)
# print(y)

import moex as mx
import requests

with requests.Session() as session:
    bb = mx.get_board_history(session, security='RU000A0ZZES2', start='2019-10-21', end='2019-10-21',
                              board=board2, market='bonds')
    aa = mx.get_market_history(session, security='TRUR', start='2020-12-15', end='2020-12-15',
                               market='shares')
    cc = mx.find_security_description(session, security='TECH')
    dd = mx.get_board_securities(session)
    print(bb[0])
    print(aa)
    print(pd.DataFrame(dd))
