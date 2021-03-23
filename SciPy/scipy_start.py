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
from scipy.stats import gamma
lambda_g = 1
k = 1
x = np.linspace(0, 30, 200)  # задаем СВ Х
gamma_distribution = gamma(lambda_g, 0, k)  # 0 - это смещение
# mean = gamma_distribution.mean()  # вывод (также std,
pdf = gamma_distribution.pdf(x)  # значения функции плотности вероятности
# sample = gamma_distribution.rvs(size=5)  # значения значения плотности вероятности (5 штук)
df = pd.DataFrame({'x': x, 'y': gamma_distribution.pdf(x)})
fig = px.line(df, x=x, y='y')
fig.show()


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


# Критерий Манна-Уитни
# from scipy.stats import mannwhitneyu
# u, p = mannwhitneyu(x, y)


# Расстояние хи-квадрат Пирсона
# a = chisquare([795, 705])  # вывод Power_divergenceResult(statistic=5.4, p-value=0.02013675155034633)
# Cumulative distribution function (cdf) - ФР
# a = scipy.stats.chi2.cdf(2, 2)  # p-value при df=2, со значением хи2=2
# более подробная информация
# observed = [[10, 6], [5, 15]]
# chi2, p, df, exp = chi2_contingency(observed,
#                                     correction=False) - поправка Йетса

# Точный критерий фишера
# observed = [[1, 3], [3, 1]]
# result = scipy.stats.fisher_exact(observed)

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

# разделение данных на обучающую выборку (training set) и контрольную (testing set)
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     random_state=42,  # управляет перемешиванием
#                                                     train_size=0.33)  # размер тренировочной выборки


# Выбор модели


# линейная регрессия
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# загрузка данных
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
df_train['sex_cod'] = LabelEncoder().fit_transform(df_train.Sex)
# print(df_train)
# x_train = df_train.sex_cod
# y_train = df_train.Survived
#
# logreg = LogisticRegression()
# model = logreg.fit(x_train, y_train)
# print(model)

# offline.plot(create_histogram())