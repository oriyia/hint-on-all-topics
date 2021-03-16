from scipy.stats import chi2_contingency
from scipy.stats import chisquare
import scipy
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline as offline

# задание нормального распределения
np.random.seed(0)  # - для какой-то там нормализации
x_rand1 = np.random.normal(loc=0, scale=1, size=1000)
x_rand2 = np.random.normal(loc=0, scale=1, size=1000) + 0.5
# Группировка данных вместе
hist_data = [x_rand1, x_rand2]  # для нормальной работы, когда одна СВ. Список списков
# Задание цвета
colors_x3 = ['#835AF1', '#7FA6EE', '#B8F7D4']
colors_x2 = ['#835AF1', '#7FA6EE']


def create_histogram():
    fig = px.histogram(x_rand1,
                       nbins=100,  # количество разбиений
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


# Расстояние хи-квадрат Пирсона
# a = chisquare([795, 705])  # вывод Power_divergenceResult(statistic=5.4, p-value=0.02013675155034633)
# Cumulative distribution function (cdf) - ФР
# a = scipy.stats.chi2.cdf(2, 2)  # p-value при df=2, со значением хи2=2


# offline.plot(create_distplot())
