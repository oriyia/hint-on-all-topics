# МАШИННОЕ ОБУЧЕНИЕ - создание моделей данных. (построение математических моделей данных).
# столбцы - это признаки (матрица признаков).
# размерность - [выборки, признаки]

# целевой массив - одномерен, столбец Series
import pandas as pd
import numpy as np

"""
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
"""

# КОДИРОВКА ПЕРЕМЕННЫХ
# from sklearn.preprocessing import LabelEncoder
# codes = LabelEncoder().fit(df_train.Sex)  # создаем экземпляр класса и кодируем
# print(codes.classes_)  # вывод этикеток кодировки ['female' 'male']
# a = codes.transform(df_train.Sex)  # и трансформировать входные данные объекта Series в соответствии с кодами


# разделение данных на обучающую выборку (training set) и контрольную (testing set)
from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                     random_state=42,  # управляет перемешиванием
#                                                     train_size=0.33)  # размер тренировочной выборки


# Выбор модели


# линейная регрессия
# from sklearn.linear_model import LinearRegression  # выбираем класс модели
# from sklearn.datasets import load_boston
# boston = load_boston()  # создаем экземпляр модели
# features = boston.data[:, 0:2]  # только 2 предиктора (признака)
# target = boston.target
# # создать объект (экземпляр класса) линейной регрессии
# regression = LinearRegression()
# подгонка линейной регрессии
# model = regression.fit(X_train, y_train)
# print(model.intercept_)  # вывод intercept
# print(model.coef_)  # вывод коэффициентов
# предсказание
# y_pred = model.predict(X_test)


# Гауссов наивный байесовский классификатор
from sklearn.naive_bayes import GaussianNB  # выбираем класс модели
# model = GaussianNB()  # создаем экземпляр класса
# model.fit(x_train, y_train)  # обучение модели
# y_model = model.predict(x_test)  # предсказание


# ПОНИЖЕНИЕ РАЗМЕРНОСТИ
# 1) МЕТОД ГЛАВНЫХ КОМПОНЕНТ
# from sklearn.decomposition import PCA  # выбор класс модели
# model = PCA(n_components=2)  # создание экземпляра класса с гиперпараметрами
# model.fit(x_train)  # обучение модели на данных, у мы не указываем
# x_2d = model.transform(x_train)  # преобразуем данные в двумерные

# КЛАСТЕРИЗАЦИЯ
# 1) Смесь Гауссовых распределений
# from sklearn.mixture import GaussianMixture
# model = GaussianMixture(n_components=3, covariance_type='full')  # создаем экземпляр с гиперпараметрами
# model.fit(x_train)  # обучаем модель, у не указываем
# y_gmm = model.predict(x_test)  # определяем метки классов


# Оценка эффективности модели (оценка точности)
from sklearn.metrics import accuracy_score
# 1 способ - обычная проверка
# score = accuracy_score(y_true=y_test, y   _pred=y_pred)
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
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
# df_train = pd.read_csv('train.csv')
# df_train['sex_cod'] = LabelEncoder().fit_transform(df_train.Sex)  # кодировка переменной (0, 1)
# x_train = df_train.sex_cod.values.reshape(-1, 1)  # преобразование списка в список списков
# y_train = df_train.Survived
# logreg = LogisticRegression()
# model = logreg.fit(x_train, y_train)
# print(model.coef_)
# print(model.intercept_)
# offline.plot(create_histogram())
