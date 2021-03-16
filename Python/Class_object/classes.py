# КЛАССЫ
# наследование, композиция
# суперклассы - подклассы - экземпляры
# образуется дерево поиска в иерархии наследования
# class оператор
# аргумент self пердоставляет возможность обращаться к обрабатываемому экземпляру
# возможность настройки, путем создания новых подклассов, вместо изменения кода
# класс, из которого создается экземпляр, определяет уровень, откуда будет начинаться поиск атрибутов
# при каждом обращении к классу, он создает новый объект экземпляра
# ИНКАПСУЛЯЦИЯ - написание кода один раз в классе, и многократное его использование в будущем
# наследование, настройка, расширение --> сокращение избыточности
# ПЕРЕГРУЗКА ОПЕРАЦИЙ - перехват встроенных операций в методах класса

class FirstClass:
    n = 'adfas'
    def setdata(self, value):
        self.data = value
    def display(self):
        print(self.data)
class SecondClass(FirstClass):
    def display(self):
        print('Current value: "%s"' % self.data)

# создание экземпляров
x = FirstClass()
y = FirstClass()

x.setdata('afasdf')
x.display()

x.data = 'assss'
x.display()
# генерация нового атрибута в пространстве имен экземпляра
x.anothername = 'sdfsd'

z = SecondClass()
z.setdata('none')
z.display()

# __init__ - метод конструктора

class ThirdClass(SecondClass):
    def __init__(self, value):
        self.data = value
    def __add__(self, other):
        return ThirdClass(self.data + other)
    def __str__(self):
        return '[ThirdClass: %s]' % self.data
    def mul(self, other):
        self.data *= other

a = ThirdClass('abc')
a.display()

# print(a)  # [ThirdClass: abc]

b = a + 'xyz'
# print(b)  # [ThirdClass: abcxyz]

a.mul(3)
# print(a)  # [ThirdClass: abcabcabc]

# атрибуты объектов располагаются в словарях
# print(list(FirstClass.__dict__.items()))  # [('__module__', '__main__'), ('x', 2), ('setdata', <func

# простая функция за пределами класса
def upperfun(obj):
    return obj.data.upper()
# print(upperfun(x))  # SDFSD

FirstClass.method = upperfun  # теперь это метод класса
# print(z.method())  # NONE
# print(FirstClass.method(z))  # NONE

class Person:

    def __init__(self, name, job=None, pay=0):
        self.name = name
        self.job = job
        self.pay = pay

    def info(self):
        return (self.name, self.job)

    def lastname(self):
        return self.name.split()[-1]

    def giveraise(self, percent):
        self.pay = int(self.pay * (1 + percent))

    def __repr__(self):
        return '[Person: %s, %s, %s]' % (self.name, self.job, self.pay)


class Manager(Person):

    def __init__(self, name, pay):
        Person.__init__(self, name, 'mng', pay)

    def giveraise(self, percent, bonus=.10):  # настройка засчет переопределения
        Person.giveraise(self, percent + bonus)

    def somethingelse(self):  # расширение
        pass


# код для тестирования
if __name__ == '__main__':
    bob = Person('Bob Smith')
    sue = Person('Sue Jones', job='def', pay=100000)
    # print(bob.name, bob.pay)
    # print(sue.name, sue.pay)
    # print(bob.lastname(), sue.lastname())
    sue.giveraise(.01)
    # print(sue)
    tom = Manager('Tom Jones', pay=50000)
    tom.giveraise(.10)
    # print(tom)
    # вывод всех сразу
    for obj in (bob, sue, tom):
        obj.giveraise(.10)
        # print(obj)
    # print(getattr(bob, 'name'))


class AttrDisplay:
    """
    fdfsdfjsdalfkjdaslf
    asdfjadsfasdfasdfdfadsf jkd klsdf lajsd f
    """

    def gatherattr(self):
        attrs = []
        for key in sorted(self.__dict__):
            attrs.append('%s=%s' % (key, getattr(self, key)))
        return ', '.join(attrs)

    def __repr__(self):
        return '[%s: %s]' % (self.__class__.__name__, self.gatherattr())


if __name__ == '__main__':
    class TopTest(AttrDisplay):
        count = 0

        def __init__(self):
            self.attr1 = TopTest.count
            self.attr2 = TopTest.count + 1
            TopTest.count += 2

    class SubTest(TopTest):
        pass

x, y = TopTest(), SubTest()
# print(x)  # [TopTest: attr1=0, attr2=1]
# print(y)  # [SubTest: attr1=2, attr2=3]
# print(AttrDisplay.__name__)  # AttrDisplay
# print(AttrDisplay.__doc__)  # вывод строк документации

# class _artist: or class __artist: - псевдозакрытые атрибуты класса



# ИСКЛЮЧЕНИЕ
# ИЗ-ЗА МЕТОДА delegate класс SUPER является абстрактным супер классом
class Super:
    def delegate(self):
        self.action()
    def action(self):
        assert False, 'action must be defined'
        # или raise NotImplementedError('action must be defined')
# для устранения ошибки необходимо в подклассах определить абстрактные методы

# область видимости всегда определяестя местоположением присваиваний в исходном коде,
# и никогда не зависит от того, что и куда импортируется



# ОБЛАСТИ ВИДИМОСТИ ДЛЯ КЛАССОВ И ФУНКЦИЙ
def afaaf():
    o = 1
    class As:
        print(o)  # классы имеют доступ к областям видимости объемлющих функций
        p = 2
        def dddd(self):
            print(As.p)  # чтобы получилось, нужно делать вот так (но значание для экземпляра)
            print(self.p)  # или так (как атрибуты объекта класса или экземпляра
            # print(p) ошибка! но вложенный код в класс не имеет доступа к объемлющей области этого класса
    # print(As.p) так будет работать
# print(afaaf())



# ЭВОЛЮЦИЯ СЛОВАРЯ ПРОСТРАНСТВА ИМЕН ДЛЯ ЭКЗЕМПЛЯРА
class Super:
    def hello(self):
        self.data1 = 'spam'
class Syo:
    pass
class Mo(Super, Syo):
    def yhy(self):
        self.data2 = 'nono'

w = Mo()
# print(w.__dict__)  # {} при создании экза он пуст
w.hello()
# print(w.__dict__)  # {'data1': 'spam'} атрибуты оказываются в словарях пространств имен экземпляров
# print(w.__class__)  # <class '__main__.Mo'> инфа о принадлежности к классу
# print(Mo.__bases__)  # (<class '__main__.Super'>, <class '__main__.Super'>) инфа о принадлежности к суперклассам
# print(Mo.__base__)  # <class '__main__.Super'>
# print(Super.__bases__)  # (<class 'object'>,)
# print(list(Mo.__dict__.keys()))  # ['__module__', 'yhy', '__doc__']
aw = 'data1'
# print(w.aw)  # будет ошибка, такого атрибута нет в объекте Mo
# print(w.__dict__[aw])  # 'spam' тут уже все норм
# print(dir(w))  # dir уже покажет также все атрибуты в том числе
# и унаследованные holla и yhy ['__class__', '__delattr__', ...]
# print(Super.__base__.__base__)  # None

# def __init_(self): - для вновь созданного экза появляются на автомате атрибуты (т.к __init на автомате выполняется)

class ma:
    x = 1
    def __init__(self):
        self.name = 'ldsafs'  # в классе нет этого атрибута

# print([name for name in ma.__dict__ if not name.startswith('__')])  # ['x'] нет name



# НАРЕЗАНИЕ __GETITEM__ AND ПРИСВАИВАНИЕ ПО ИНДЕКСУ: __SETITEM__
class Get:
    data = [1, 2, 3, 4, 5]
    def __repr__(self):
        return '%s' % self.data
    def __setitem__(self, item, value):
        self.data[item] = value
        return self.data
    def __getitem__(self, item):
        print('getitem:', item)
        print('slice:', item.start, item.stop, item.step)
        return self.data[item]
t = Get()
# print(t[::2])  # getitem: slice(None, None, 2)   [1, 3, 5]
# print(t[1:5])  # slice: 1 5 None
# t[0] = 10  # [10, 2, 3, 4, 5]

class aaa:
    def __getitem__(self, i):
        return self.data[i]

r = aaa()
r.data = 'spam'
# for x in r:
#     print(x, end='')  # 'spam'
# print([x for x in r])  # ['s', 'p', 'a', 'm']
# print('p' in r)  # True
# print(list(map(str.upper, r)))  # ['S', 'P', 'A', 'M']
(a, b, c, d) = r
# print(a, c, d)  # s a m

# НО ЛУЧШЕ использовать __iter__
class Other:
    def __init__(self, start, stop):
        self.value = start - 1
        self.stop = stop
    def __iter__(self):
        return self  # получить объект итератора при вызове iter (одна копия состояния итерации)
    def __next__(self):
        if self.value == self.stop:
            raise StopIteration
        self.value += 1
        return self.value ** 2

# for x in Other(1, 5):
#     print(x, end=' ')  # 1 4 9 16 25 - только один раз вывод для данного экза

# x = Other(1,5)
# print(tuple(x), tuple(x))  # (1, 4, 9, 16, 25) () только одни раз вывод (одна копия)
# x = list(Other(1,5))
# print(tuple(x), tuple(x))  # (1, 4, 9, 16, 25) (1, 4, 9, 16, 25) список поддерживает множество просмотров


# ПРИМЕНЕНИЕ КЛАССОВ И ГЕНЕРАТОРНЫХ ФУНКЦИЙ YIELD
class New:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
    def __iter__(self):
        for i in range(self.start, self.stop + 1):
            yield i ** 2

# print([x for x in New(1, 4)], end=' ')  # [1, 4, 9, 16]


# ПОДДЕРЖКА МНОЖЕСТВА ИТЕРАТОРОВ
I = New(1, 3)  # создание экза
for i in I:
    for j in I:
        pass
        # print('%s:%s' % (i, j), end=' ')  # 1:1 1:4 1:9 4:1 4:4 4:9 9:1 9:4 9:9


# ДОСТУП К АТРИБУТАМ: __GETATTR__, __SETATTR__

# СЛОЖЕНИЕ __ADD__ AND __RADD__
class rdd:
    def __init__(self, val=None):
        self.val = val
    def __add__(self, other):
        if isinstance(other, rdd):
            other = other.val  # создание атрибута у объекта
        return rdd(self.val + other)
    __radd__ = __add__
    def __str__(self):
        return 'To: %s' % self.val
    def __iadd__(self, other):
        self.val += other
        return self
    def __call__(self, *args, **kwargs):
        return 'Called: %s %s' % (args, kwargs)
x = rdd(2)
y = rdd(3)
# print(x+y)  # 5
z = x + y
# print(x+y)  # To: 5
# СЛОЖЕНИЕ НА МЕСТЕ __iADD__
u = rdd(2)
u += 2
# print(u)  # To: 4
i = rdd([1])
i += 2, 3, 4
# print(i)  # To: [1, 2, 3, 4]
# print(i.val)  # [1, 2, 3, 4]
# ВЫЗОВЫ __CALL__
p = rdd(2)  # self.val = 2
# print(p(1, 2, 3, x=1, y=2))  # Called: (1, 2, 3) {'x': 1, 'y': 2}


# ПСЕВДОЗАКРЫТЫЕ АТРИБУТЫ
class a1:
    def meth(self):
        self.__x = 10
class a2:
    def math(self):
        self.__x = 12
class a3(a1, a2):
    pass
i = a3()
i.meth()
i.math()
# print(i.__dict__)  # {'_a1__x': 10, '_a2__x': 12}
# АНАЛОГИЧНО И ДЛЯ ФУНКЦИЙ
# def __mean():


# СВЯЗАННЫЕ И НЕСВЯЗАННЫЕ МЕТОДЫ
class sv:
    def __init__(self, arg=None):
        self.arg = arg
    def pr(self, message):
        print(message)
    def por(self):
        return 2 * self.arg
    def prr(arg):  # простая функция
        print(arg)
# СВЯЗАННЫЙ МЕТОД - ЭТО ОБЪЕДИНЕНИЕ ФУНКЦИИ С ЭКЗЕМПЛЯРОМ В ЕДИНЫЙ ОБЪЕКТ
t = sv(2)  # создали экз
g = t.pr  # ссылка на метод экза
h = sv(3)  # еще одни экз

# l = [t.por, h.por]  # список связанных методов
# for x in l:
#     print(x(), end=', ')  # 4, 6,

# print(g('ghghg'))
# print(t.prr('af'))  # выдаст ошибку, т.к передается лишний агрумент (2 вместо 1)
# НЕСВЯЗАННЫЙ МЕТОД
t = sv()  # экз
g = sv.pr  # ссылка на метод класса
# print(g(t, 'aaaaa'))
# но несвязанный метод - это простая функция без self (например def pr(arg1, arg2))


# ФАБРИКИ ОБЪЕКТОВ
def factory(aClass, *args, **kwargs):
    return aClass(*args, **kwargs)
class Spam:
    def doit(self, message):
        print(message)
class Person:
    def __init__(self, name, job=None):
        self.name = name
        self.job = job
object1 = factory(Spam)
object2 = factory(Person, 'Ivan', 'progr')
object3 = factory(Person, name='Ilya')
# print(object1.doit(99))  # 99
# print(object2.name, object2.job)  # Ivan progr
# print(object3.name)  # Ilya


# МНОЖЕСТВЕННОЕ НАСЛЕДОВАНИЕ: ("ПОДМЕШИВАЕМЫЕ КЛАССЫ")


# МЕТОД SUPER
class rob:
    xaxa = 2

    def __init__(self, name):
        self.name = name

    def mono(self):
        print('робот езид по кругу')


class rob2:

    def __init__(self, height):
        self.height = height


class rob1(rob, rob2):

    def __init__(self, name, gun, height):
        rob2.__init__(self, height)
        rob.__init__(self, name)
        # super().__init__(name=name)  # можно и вот так
        super().mono()
        self.gun = gun

x = rob1('asa', 'pist', 89)
y = rob('aha')
l = list(y.__dict__.keys())
print(l, 'xaxa' in list(y.__dict__.keys()))