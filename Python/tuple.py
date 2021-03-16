# t = (1, 2, 3, 4) -- (1, 2, 3, 4)

# t = 1, 2, 'dfdf', [3, 4, 5]  # --  (1, 2, 'dfdf', [3, 4, 5]) альтернативный способ создания кортежа

# t = tuple('asdfa')

# t2 = (2,) + t[2:] -- (2, 'dfdf', [3, 4, 5])

# print(t[3][2]) -- 5

# t = (9,) -- запятая обязательна, иначе не кортеж, а просто число

# sorted(t) -- сортировка

# t[3][0] = 'yxti' -- (1, 2, 'dfdf', ['yxti', 4, 5]) - изменяемые объекты внутри кортежа можно изменять

# МОДУЛЬ COLLECTIONS - ИМЕНОВАННЫЙ КОРТЕЖ
d = dict(t=1, y=3)
print(d)  # -- {'t': 1, 'y': 3}
from collections import namedtuple
tup = namedtuple('tup', d.keys())
bob = tup(*d.values())
# print(bob) -- tup(t=1, y=3)
# print(bob[0]) -- 1
# print(bob.t) -- 1
#
# o = bob._asdict() -- перевод обратно в словарь
# print(o) -- {'t': 1, 'y': 3}

t, y = bob
print(t)