# WHILE

# while True: -- бесконечный цикл
#     print('Type Ctrl-C to stop me!')

# for

# for ((a, b), c) in [([1, 2], 3), ['xy', 6]]: print(a, b, c)
# S = 'aadfaddfafaf'
# for c in S[::2]: print(c, end=' ') -- a d a d a a

# [x+1 for x in L] -- списковое включение

# for (x, y) in zip(L1, L2):

keys = ['spam', 'eggs']
vals = [1, 2]
# print(dict(zip(keys, vals))) -- {'spam': 1, 'eggs': 2}

# a = {k: v for (k, v) in zip(keys, vals)} -- {'spam': 1, 'eggs': 2}

# for (a, b) in enumerate(keys): 0 spam, 1 eggs
# for (i, s) in enumerate(open('text.txt')):

