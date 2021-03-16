# print(list(bool(x) for x in ('', 0, None))) -- [False, False, False] все остальное True

# print(2 or 3) -- 2 булевские операции возвращают объект
# print(3 or 2) -- 3
# print([] or 3) -- 3
# print([] or {}) -- {}
# print(2 and 3) -- 3
# print([] and {}) -- []

# x = 0; y = 2; z = 3
# a = y if x else z -- 3
# print(a)

a = [1, 2, '', 3, 4, [], 0]
# print(a[bool()]) --
# print(list(filter(bool, a))) -- [1, 2, 3, 4]
# print([x for x in a if x]) -- [1, 2, 3, 4]
# print(any(a), all(a)) -- True False
