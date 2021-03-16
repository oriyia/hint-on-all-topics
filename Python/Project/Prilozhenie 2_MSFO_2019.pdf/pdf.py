class Fraction:

    def __init__(self, a, b=None, op=None):
        self.a = a
        self.op = op
        if b:
            self.b = b
        else:
            self.b = 1

    def __add__(self, other):
        op = 'сложение'
        if isinstance(other, Fraction):
            other1 = other.a
            other2 = other.b
            new_a = self.a * other2 + other1 * self.b
            new_b = self.b * other2
            return Fraction(new_a, new_b, op)
        else:
            new_a = self.a + other * self.b
            new_b = self.b
            return Fraction(new_a, new_b, op)
    __radd__ = __add__

    def __sub__(self, other):
        op = 'вычитание'
        if isinstance(other, Fraction):
            other1 = other.a
            other2 = other.b
            new_a = self.a * other2 - other1 * self.b
            new_b = self.b * other2
            return Fraction(new_a, new_b, op)
        else:
            new_a = self.a - other * self.b
            new_b = self.b
            return Fraction(new_a, new_b, op)
    __rsub__ = __sub__

    def __mul__(self, other):
        op = 'умножение'
        if isinstance(other, Fraction):
            other1 = other.a
            other2 = other.b
            new_a = self.a * other1
            new_b = self.b * other2
            return Fraction(new_a, new_b, op)
        else:
            new_a = self.a * other
            new_b = self.b
            return Fraction(new_a, new_b, op)
    __rmul__ = __mul__

    def __repr__(self):
        if self.a == self.b:
            return 'Операция \'%s\': %s' % (self.op, 1)
        elif self.a % self.b == 0:
            return 'Операция \'%s\': %s' % (self.op, self.a / self.b)
        else:
            # Сокращение дроби. Поиск наибольшего общего делителя для числителя и знаменателя
            n = max(i for i in range(1, max(self.b, self.a)) if self.a % i == 0 and self.b % i == 0)
            a = int(self.a / n)
            b = int(self.b / n)
            return 'Операция \'%s\': %s/%s' % (self.op, a, b)


class OperationsOnFraction(Fraction):

    def getint(self):
        number = int(self.a / self.b)
        return number

    def getfloat(self):
        number = float(self.a / self.b)
        return number


if __name__ in '__main__':
    x = Fraction(3, 5)
    y = Fraction(4, 7)
    z = Fraction(3)
    print(x + y)
    print(z + y)
    print(z - x)
    print(x - z)
    print(x * y)
    l = OperationsOnFraction(3, 5)
    m = OperationsOnFraction(4, 7)
    p = l + m
    print(p, OperationsOnFraction.getint(p), OperationsOnFraction.getfloat(p), sep=', ')
    print(Fraction(2, 3) + Fraction(1, 3))
    print(Fraction(2) + Fraction(1))
    df = Fraction(2)
    dc = Fraction(3)
    dd = dc + df
    print(dd, OperationsOnFraction.getint(dd), OperationsOnFraction.getfloat(dd), sep=', ')
    print(1 + l)
    print(1 - l)
    k = 2 * l
    print(k, OperationsOnFraction.getint(k), OperationsOnFraction.getfloat(k), sep=', ')



