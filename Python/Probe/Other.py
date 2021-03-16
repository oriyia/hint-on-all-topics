# def div():
#     for i in range(2):
#         x = int(input("enter a number: "))
#         y = int(input("enter another number: "))
#         print(x, '/', y, '=', x/y)
# try:
#     div()
# except ZeroDivisionError as error:
#     print(f'Ошибка \'{error}\'! На ноль делить нельзя.')
# except ValueError as error:
#     print(f'Ошибка \'{error}\'! Вы должны ввести целое число.')
# finally:
#     print('Конец!')
#
#
# # def sumOfPairs(L1, L2):
# #     sum = 0
# #     sumOfPairs = []
# #     for i in range(len(L1)):
# #         sumOfPairs.append(L1[i] + L2[i])
# #
# #     print("sumOfPairs = ", sumOfPairs)
# #
# #
# # try:
# #     sumOfPairs('a3')
# #
# # except IndexError as x:
# #     print('Длина первой строки не должна быть больше длины второй.')
# # except TypeError as x:
# #     if 'sumOfPairs()' in x.args[0]:
# #         print('Разрешено передавать функции только два аргумента')
# #     else:
# #         print('В качестве агрумента должна быть передана строка')



class NotNameError(Exception):
    pass


class NotEmailError(Exception):
    pass


def red(line):
    name, email, age = line.split()
    age = int(age)
    if not name.isalpha():
        raise NotNameError
    elif email.find('@') == -1 or email.find('.') == -1:
        raise NotEmailError
    elif 10 > age > 90:
        raise ValueError
    else:
        good.write(line + '\n')


with open('registrations_.txt', 'r', encoding='utf-8') as file, \
        open('registrations_bad.log', 'w') as bag, open('registrations_good.log', 'w') as good:
    for line in file:
        line = line[:-1]
        try:
            red(line)
        except ValueError as exc:
            if 'too' in exc.args[0]:
                bag.write(line + ' ValueError: Слишком много полей, необходимо только 3.\n')
            elif 'not' in exc.args[0]:
                bag.write(line + ' ValueError: Не присутствуют все три поля.\n')
            else:
                bag.write(line + ' ValueError: Поле "возраст" не является числом от 10 до 99.\n')
        except NotNameError:
            bag.write(line + ' NotNameError: поле имени содержит не только буквы\n')
        except NotEmailError:
            bag.write(line + ' NotEmailError: поле email не содержит @ и .(точку)\n')

with open('registrations_good.log', 'r') as good, open('registrations_bad.log', 'r') as bad:
    for i, line in enumerate(good):
        print(line, end='')
        if i == 5:
            break
    for i, line in enumerate(bad):
        print(line, end='')
        if i == 5:
            break


