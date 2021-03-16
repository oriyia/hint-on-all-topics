m = [[1, 2, 3],[3, 2, 3],[4, 2, 3]]
# cor = [row[0] + 1 for row in m if row[1] % 2==0]
# #print(cor)

# g = (sum(row) for row in m) - генератор сумм элементов в строках
# print(next(g))
# print(next(g))

# print(list(map(sum, m))) -- 6, 8 ,9

# print({sum(row) for row in m}) #-- сумма элементов строк, но они записываются вразнобой{8, 9, 6}

# print({i:sum(m[i]) for i in range(3)}) -- {0: 6, 1: 8, 2: 9}

# help(list) - список всех доступных методов для данного объекта


