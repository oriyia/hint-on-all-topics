#%%
from scipy.stats import chi2_contingency
from scipy.stats import chisquare
import scipy
import kaleido
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline as offline
import sklearn
import pandas as pd
import psutil
import plotly.graph_objs as go
pd.set_option('display.max_columns', 14)


# задание нормального распределения
np.random.seed(0)  # - для какой-то там нормализации
x_rand1 = np.random.normal(loc=0, scale=1, size=1000)
x_rand2 = np.random.normal(loc=0, scale=1, size=1000) + 0.5
# Группировка данных вместе
hist_data = [x_rand1, x_rand2]  # для нормальной работы, когда одна СВ. Список списков
# Задание цвета
colors_x3 = ['#835AF1', '#7FA6EE', '#B8F7D4']
colors_x2 = ['#835AF1', '#7FA6EE']


# РАСПРЕДЕЛЕНИЯ
# ГАММА РАСПРЕДЕЛЕНИЕ
from scipy.stats import gamma
lambda_g = 1
k = 1
x = np.linspace(0, 30, 200)  # задаем СВ Х
gamma_distribution = gamma(lambda_g, 0, k)  # 0 - это смещение
# mean = gamma_distribution.mean()  # вывод (также std,
pdf = gamma_distribution.pdf(x)  # значения функции плотности вероятности
# sample = gamma_distribution.rvs(size=5)  # значения значения плотности вероятности (5 штук)
# stats = gamma_distribution.stats(1) ТАК А КАКИЕ ТАМ ЕЩЕ ЕСТЬ АРГУМЕНТЫ
df = pd.DataFrame({'x': x, 'y': pdf})
fig = px.line(df, x=x, y='y')
# ПРИМЕР С ОБУЧЕНИЕМ
# df = pd.read_csv('OnlineRetail.csv')
# df = df[(df.UnitPrice > 0) & (df.Quantity > 0)]
# df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
# user_spending = df.groupby(['CustomerID'])['TotalPrice'].sum()
# user_spending = user_spending[user_spending.values < 10000]
# params = gamma.fit(user_spending.values)
# line = np.linspace(10, 10000, 200)
# fig = px.histogram(user_spending, nbins=50, histnorm='probability density')
# fig.add_scatter(x=line, y=gamma(*params).pdf(line))
# fig.show()

# НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ
from scipy.stats import norm
# norm_distribution = norm()

# ПРОВЕРКА РАСПРЕДЕЛЕНИЯ НА НОРМАЛЬНОСТЬ
# №1 QQ-PLOT
# from statsmodels.graphics.gofplots import qqplot
# df = pd.read_csv('weight-height.csv')
# qqplot_data = qqplot(df.Height, line='s').gca().lines
# fig = px.line(x=qqplot_data[1].get_xdata(), y=qqplot_data[1].get_ydata())
# fig.add_scatter(x=qqplot_data[0].get_xdata(), y=qqplot_data[0].get_ydata(), mode='markers', name='Норма')
# fig.show()

# №2 Критерий Шапиро-Уилка
# from scipy.stats import shapiro
# shapiro_test = shapiro(df.Height)
# statistic, p_value = shapiro(df.Height)  # 0.9960623383522034 8.162489641544413e-16
# print(shapiro_test)  # ShapiroResult(statistic=0.9960623383522034, pvalue=8.162489641544413e-16)


def create_histogram():
    fig = px.histogram(x_rand1,
                       nbins=100,  # количество разбиений
                       histnorm='probability density',  # тип нормализации гистограммы (здесь плотность вероятности)
                       cumulative=True,  # кумулятивная функция
                       opacity=0.8,  # насыщенность графика
                       marginal='box')  # дополнительное частотное распределение возле графика (rug - насечки)
    return fig


def create_distplot():
    fig = ff.create_distplot(hist_data,
                             ['name_group1', 'name_group2'],
                             colors=colors_x2,  # изменение цвета графика
                             curve_type='normal',  # тип изгиба кривой (kde=true)
                             # show_hist=False,  # не показывать гистограмму
                             # show_curve=False,  # не показывать кривую
                             show_rug=False,
                             bin_size=[.1, .6])  # количество разбиений, для каждой группы отдельно
    return fig


"""-----------------------------------------------------------------------------------------"""
"""-----------------------------------------------------------------------------------------"""
"""СТАТИСТИЧЕСКИЕ КРИТЕРИИ"""

""" 1) Параметрические"""
# однофакторный дисперсионный анализ
# from scipy.stats import f_oneway
# tillamook = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735, 0.0659, 0.0923, 0.0836]
# newport = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725]
# petersburg = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
# magadan = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764, 0.0689]
# tvarminne = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]
# result = f_oneway(tillamook, newport, petersburg, magadan, tvarminne)

# 2-х факторный дисперсионный анализ
# import pingouin as pg
# dv - это название столбца, содержащего зависимую переменную,
# between это столбец, содержащий коэффициент между группами.
# aov = pg.anova(dv='Result', between=['Rambler', 'Design'], data=data, detailed=True)
# print(aov)


""" 2) Непараметрические"""
"""Зависимые выборки"""
# критерий Уилкоксона
# from scipy.stats import wilcoxon
# разница между двумя величинами
# d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]
# w, p = wilcoxon(d)
# либо x и y указываем
# w, p = wilcoxon(x, y)

# критерий Фридмана
# from scipy.stats import friedmanchisquare
# result = friedmanchisquare(measurements1, measurements2, measurements3)

"""Независимые выборки"""
# Критерий Манна-Уитни
# from scipy.stats import mannwhitneyu
# u, p = mannwhitneyu(x, y)

# критерий Краскела-Уоллиса
# from scipy.stats import kruskal
# x = [1, 1, 1]
# y = [2, 2, 2]
# z = [2, 2]
# result = kruskal(x, y, z)  # KruskalResult(statistic=7.0, pvalue=0.0301973834223185)

""" 3) Номинативные переменные"""
# Расстояние хи-квадрат Пирсона
# a = chisquare([795, 705])  # вывод Power_divergenceResult(statistic=5.4, p-value=0.02013675155034633)
# Cumulative distribution function (cdf) - ФР
# a = scipy.stats.chi2.cdf(2, 2)  # p-value при df=2, со значением хи2=2
# более подробная информация
# observed = [[10, 6], [5, 15]]
# chi2, p, df, exp = chi2_contingency(observed,
#                                     correction=False) - поправка Йетса

# Критерий Мак-Немара
# from statsmodels.stats.contingency_tables import mcnemar
# # define contingency table
# table = [[4, 2],
#          [1, 3]]
# result = mcnemar(table, exact=True)
# print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

# Критерий Кохрена Q
from statsmodels.stats.contingency_tables import cochrans_q

# Таблица сопряженности
# table = pd.crosstab(df_train.Survived, df_train.Sex)


# Точный критерий фишера
# observed = [[1, 3], [3, 1]]
# result = scipy.stats.fisher_exact(observed)


# BOOTSTRAP
# сравни как работают оба метода, и какой быстрее в итоге
# import scikits.bootstrap as bootstrap
# from matplotlib import pyplot as plt
# data = stats.poisson.rvs(33, size=15000)
# results = bootstrap.ci(data=data, statfunction=scipy.mean)
# print(results)

# еще метод
# import bootstrapped.bootstrap as bs
# import bootstrapped.stats_functions as bs_stats
# mean = 354
# stdev = 20
# population = np.random.normal(loc=mean, scale=stdev, size=15000)
# samples = population[:2000]
# print(bs.bootstrap(samples, stat_func=bs_stats.mean))
# print(bs.bootstrap(samples, stat_func=bs_stats.std))


def get_bootstrap_sample(x, B_sample=1):
    """
    x - выборка
    B_sample - сколько бутстреповских выборок нужно делать в конечном итоге
    """
    N = x.size  # размер выборки
    sample = np.random.choice(x, size=(N, B_sample), replase=True)  # c повторениями, размером х на B_sample

    if B_sample == 1:
        sample = sample.T[0]  # для удобства разворачиваем в вектор
    return sample


x_boot = get_bootstrap_sample(x, B_sample=10**6)  # получаем матрицу размером сто на миллион
x_boot_m = np.mean(x_boot, axis=0)  # средние по каждой бутстреповской выборке
# далее можно построить гистограмму для выборочных средних, имеющее нормальное распределение
# график qq-plot данный факт подтверждает
# построим доверительный интервал для средних Эфрона
alpha = 0.05
left = np.quantile(x_boot_m, alpha/2)
right = np.quantile(x_boot_m, 1 - alpha/2)
# на выходе получаем два числа

# доверительный интервал Холла (для центрированной статистики)
# alpha = 0.05
# theta_hat = np.mean(x)  # среднее значение исходной выборки
# средние отклонения значений бутстреповской выборки от среднего исходной выборки
# x_boot_m = np.mean(x_boot - theta_hat, axis=0)
# left = theta_hat - np.quantile(x_boot_m, 1 - alpha/2)
# right = theta_hat - np.quantile(x_boot_m, alpha/2)

# t-процентильный доверительный интервал
# theta_hat = np.mean(x)
# std_hat = np.std(x)
#
# x_boot_t = np.mean(x_boot - theta_hat, axis=0)
# x_boot_t = x_boot_t / np.std(x_boot, axis=0)
# left = theta_hat - np.quantile(x_boot_t, 1 - alpha/2) * std_hat
# right = theta_hat - np.quantile(x_boot_t, alpha/2) * std_hat

# доверительный интервал для разницы мужду двумя выборками (по Эфрону)
# def stat_intervals(boot, alpha=0.05):
#     left = np.quantile(boot, alpha/2)
#     right = np.quantile(boot, 1 - alpha/2)
#     return left, right
#
# stat_intervals(x_boot - y_boot)






'''
Scikit-Learn - ДОКУМЕНТАЦИЯ
Чаще всего использование API статистического оценивания библиотеки Scikit-Learn включает следующие шаги (далее мы 
рассмотрим несколько подробных примеров).
1. Выбор класса модели с помощью импорта соответствующего класса оценивателя из библиотеки Scikit-Learn.
2. Выбор гиперпараметров модели путем создания экземпляра этого класса с соответствующими значениями.
3. Компоновка данных в матрицу признаков и целевой вектор в соответствии с описанным выше.
4. Обучение модели на своих данных посредством вызова метода fit() экземпляра модели.
5. Применение модели к новым данным:
- в случае машинного обучения с учителем метки для неизвестных данных обычно предсказывают с помощью метода predict();
- в случае машинного обучения без учителя выполняется преобразование свойств данных или вывод их значений посредством 
  методов transform() или predict().
'''

# КОДИРОВКА ПЕРЕМЕННЫХ
# from sklearn.preprocessing import LabelEncoder
# codes = LabelEncoder().fit(df_train.Sex)  # создаем экземпляр класса и кодируем
# print(codes.classes_)  # вывод этикеток кодировки ['female' 'male']
# a = codes.transform(df_train.Sex)  # и трансформировать входные данные объекта Series в соответствии с кодами


# разделение данных на обучающую выборку (training set) и контрольную (testing set)
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     random_state=42,  # управляет перемешиванием
#                                                     train_size=0.33)  # размер тренировочной выборки


# Выбор модели


# линейная регрессия
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
boston = load_boston()
features = boston.data[:, 0:2]  # только 2 предиктора (признака)
target = boston.target
# создать объект (экземпляр класса) линейной регрессии
regression = LinearRegression()
# подгонка линейной регрессии
# model = regression.fit(X_train, y_train)
# print(model.intercept_)  # вывод intercept
# print(model.coef_)  # вывод коэффициентов
# предсказание
# y_pred = model.predict(X_test)


# Оценка эффективности модели (оценка точности)
from sklearn.metrics import accuracy_score
# 1 способ - обычная проверка
# score = accuracy_score(y_true=y_test, y_pred=y_pred)
# 2 способ - перекрестная проверка модели (cross validation)
from sklearn.model_selection import cross_val_score
# - Кросс-валидация по K блокам (K-fold cross-validation)
# score = cross_val_score(model, x, y, cv=5)
# score.mean()
# - Валидация последовательным случайным сэмплированием (random subsampling)
# - Поэлементная кросс-валидация (Leave-one-out, LOO)

# Кривые обучения
from sklearn.model_selection import validation_curve
# train_score, val_score = validation_curve(model(), X, y, 'parameter', degree, cv=7)
from sklearn.model_selection import learning_curve
# train_score, val_score = learning_curve(model(), X, y, 'parameter', degree, cv=7)


# Логистическая регрессия
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
df_train = pd.read_csv('train.csv')
df_train['sex_cod'] = LabelEncoder().fit_transform(df_train.Sex)  # кодировка переменной (0, 1)
x_train = df_train.sex_cod.values.reshape(-1, 1)  # преобразование списка в список списков
y_train = df_train.Survived
logreg = LogisticRegression()
model = logreg.fit(x_train, y_train)
# print(model.coef_)
# print(model.intercept_)
# offline.plot(create_histogram())


