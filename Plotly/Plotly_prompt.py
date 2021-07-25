# import plotly.plotly as py  # импорт для онлайна
import plotly.offline as offline  # импорт для оффлайна
from plotly.graph_objs import *  # импорт всех граф объекты
import plotly.express as px
# offline.init_notebook_mode()  # для использования в ноутбуке

import numpy as np
import pandas as pd

# offline.plot([{
#     'x': x,
#     'y': f(x),
# }])

import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
x = np.arange(0, 150, 1)

def f(x):
    return x**2
def h(x):
    return np.sin(x)
def g(x):
    return np.cos(x)
def j(x):
    return np.tan(x)


def t(x):
    return 0.5 * x


fig = go.Figure()  # создание поля для графика
fig = make_subplots(rows=3, cols=2, subplot_titles=('Plot1', 'Plot2', 'Plot3', 'Plot4'),
                    specs=[[{'rowspan': 3}, {}], [None, {}], [None, {}]])
# column_widths=[2, 1] задание отношения между графиками по строке, row_heights= столбцы
# specs=[[{"colspan": 2}, None], [{}, {}]]) для объединения по строке
fig.update_yaxes(range=[-0.5, 5], zeroline=True, zerolinewidth=2, zerolinecolor='LightPink',
                 col=2)  # область показа по y, причем изменяем у второго графика
fig.update_xaxes(range=[-0.5, 5], zeroline=True, zerolinewidth=2, zerolinecolor='#000',
                 col=2)  # область показа по x, аналогично

fig.add_trace(go.Scatter(x=x, y=f(x), name='f(x)=x<sup>2</sup>', mode='lines+markers'), 1, 2)
fig.add_trace(go.Scatter(x=x, y=t(x), name='f(x)=0.5x', mode='lines+markers',
                         marker=dict(color=x, colorbar=dict(title='f(x)=x'),
                                     colorscale='Inferno', size=10*t(x))), 2, 2)
fig.add_trace(go.Scatter(x=x, y=x, name='g(x)=x', mode='markers',
                         marker=dict(color='LightSkyBlue', size=10, line=dict(color='MediumPurple', width=2))), 1, 2)
fig.add_trace(go.Scatter(x=x, y=h(x), name='h(x)=sin(x)'), 3, 2)
fig.add_trace(go.Scatter(x=x, y=g(x), name='g(x)=cos(x)'), 3, 2)
# fig.add_trace(go.Scatter(visible='legendonly', x=x, y=j(x), name='j(x)=tg(x)'))  # неактивна функция, скрыта
fig.add_trace(go.Scatter(x=x, y=j(x), name='j(x)=tg(x)'), 1, 1)

fig.update_layout(legend_orientation="h",  # внизу располагается легенда
                  legend=dict(x=.5, xanchor="center"),  # выравнивание легенды (по центру)
                  margin=dict(l=0, r=0, t=60, b=0),  # настройка полей вокруг графика, сверху 30px
                  # title='Grafic',  # общее название графика
                  # xaxis_title='x Axis Title',  # подпись оси x
                  # yaxis_title='y Axis Title',  # подпись оси y
                  width=600,  # ширина
                  height=1000,  # высота
                  hovermode='x')  # при наведении на один график, данные отображаются и по второму (по иксу)
fig.update_layout(title='Plot Title')  # общее название графиков
fig.update_xaxes(title='Ось Х графика 1', col=1, row=1)  # подписывание осей
fig.update_xaxes(title='Ось Х графика 2', col=2, row=1)
fig.update_yaxes(title='Ось Y графика 1', col=1, row=1)
fig.update_yaxes(title='Ось Y графика 2', col=2, row=1)
fig.update_traces(hoverinfo='all', hovertemplate='Аргумент: %{x}<br>Функция: %{y}')
offline.plot(fig)

x_axes = [0, 1, 2]
y_axes = [0, 1, 2]

# шаблон цветовой палитры
temp_color = ['#4189c3', '#41c3a9', '#1ba672', '#6b737d', '#ffad38', '#ed5e73', '#c96dd0', '#4db2ff', '#825ec2']

# настройки темы
docs_theme = dict(
    layout=go.Layout(colorway=temp_color,  # используемая цветовая схема
                     title_font=dict(family="Helvetica",  # название графика
                                     size=28,
                                     color='#5c5c5c'),
                     legend=dict(bordercolor='#e8e8e8',  # цвет обводки
                                 borderwidth=2,  # толщина обводки
                                 font=dict(family='Helvetica',  # шрифт легенды
                                           size=22,  # размер шрифта
                                           color='#5c5c5c'),  # цвет шрифта
                                 orientation='h',  # ориентация легенды (h - горизонтальная)
                                 # x=0.7,
                                 y=-0.2),  # настройка положения относительно левого края
                     paper_bgcolor='white',  # цвет подложки изображения
                     plot_bgcolor='white',  # цвет подложки графика
                     xaxis=dict(gridcolor='#dbdbdb',  # цвет у сетки
                                gridwidth=2,  # толщина сетки
                                zerolinecolor='#b8b8b8',  # цвет оси х
                                zerolinewidth=4,  # толщина оси x
                                title=dict(font_size=23,  # подпись оси x
                                           font_color='#5c5c5c',
                                           standoff=15,  # расстояние до графика
                                           font_family='Helvetica'),
                                tickfont=dict(family='Helvetica',  # подпись тиков
                                              size=20,
                                              color='#5c5c5c'),
                                # обводка границ графика
                                showline=True,  # показать обводку
                                linewidth=1,  # толщина обводки
                                linecolor='#b8b8b8',  # цвет обводки
                                mirror=True  # полная обводка
                                ),
                     yaxis=dict(gridcolor='#dbdbdb',
                                gridwidth=2,
                                zerolinecolor='#b8b8b8',
                                zerolinewidth=4,
                                title=dict(font_size=23,
                                           font_color='#5c5c5c',
                                           font_family='Helvetica'),
                                tickfont=dict(family='Helvetica',
                                              size=20,
                                              color='#5c5c5c'),
                                showline=True, linewidth=1, linecolor='#b8b8b8', mirror=True),
                     margin=dict(l=100, r=10, t=90, b=120),
                     width=1367, height=617))  # 1367 617

fig = px.line(module_statistics_group_count,
              x=module_statistics_group_count['mean'].values,
              y=module_statistics_group_count.index.get_level_values(1),
              facet_col=module_statistics_group_count.index.get_level_values(0),
              facet_col_wrap=1,
              labels=dict(x='Длительность', color='Курс', facet_col='Курс'),
              color=module_statistics_group_count.index.get_level_values(0),
              facet_row_spacing=0.025)  # расстояние между графиками по вертикали

# просто добавить фигуру
fig.add_trace(go.Scatter(x=x_axes, y=y_axes,
                         line=dict(width=8),
                         name='beta0= -2, beta=0.8'))

# точечный график
fig.add_trace(go.Scatter(x=x_axes, y=y_axes, mode='markers',
                         marker_size=20,  # размер маркеров
                         marker_color="lightskyblue",  # цвет маркера
                         marker_line_color="midnightblue",  # цвет обводки маркера
                         marker_line_width=2,  # толщина обводки
                         marker_symbol="x-dot",  # тип маркера
                         ))

# подпись точек на графике
fig.add_trace(go.Scatter(mode="markers+text",
                         x=[4, 8],
                         y=[0.5, -0.5],
                         text=["Point A", "Point B"]))

# горизонтальная линия
fig.add_hline(y=1,  # координата через которую проходит линия
              line_dash='dash',  # тип линии (dash - пунктирная,
              line=dict(color='#8a8a8a',
                        width=5),
              layer="below")  # слой (below - на заднем плане,

fig.add_shape(type='line',  # добавить линию с указанием координат точек
              x0=3, x1=10, y0=0, y1=2,
              line=dict(width=8,
                        color='grey'))

fig.update_layout(title=dict(text='<b>Медианное время прохождения каждого модуля</b>',  # название графика
                             x=.5,
                             xanchor="center"),
                  showlegend=False,  # не показывать легенду
                  width=1200,  # ширина изображения, но не самого графика
                  height=2200,  # высота изображения
                  margin=dict(l=30, r=10, t=50, b=10),
                  template='название темы')  # использование темы оформления

fig.update_xaxes(fixedrange=True,  # размер графика по всей подложке
                 tickfont_size=9,  # размер тиков
                 showticklabels=True,  # показать все тики для express
                 ticksuffix=' дн.',  # суффикс у тиков оси
                 title=dict(text='$\Large{F_{X}}$',  # подпись оси x в Latex (\large{}, \Large{}, \huge{}, \Huge{})
                            font=dict(size=10),  # размер шрифта
                            standoff=1),  # расстояние до графика
                 nticks=6,  # количество тиков
                 tickangle=45,  # угол наклона тиков
                 tickformat='% H ~% M ~% S.% 2f',  # «2016-10-13 09: 15: 23.456» ---> «09 ~ 15 ~ 23.46»
                 range=[3, 15],  # область показа значений на графике
                 )

fig.update_yaxes(fixedrange=True,
                 tickfont_size=9,
                 side='right',  # сторона отображения тиков
                 title=dict(text='Модули',  # подпись оси у
                            font=dict(size=10)))  # размер шрифта

fig.add_annotation(x=0, y=1,  # координаты точки для подпись
                   text="Text annotation with arrow",  # сам текст подписи
                   showarrow=True,  # со стрелкой
                   font=dict(family='Helvetica',  # шрифт подписи
                             size=25,
                             color='grey'),
                   arrowwidth=3,  # толщина стрелки
                   arrowcolor='#757575',  # цвет стрелки
                   arrowsize=1,  # размер самой стрелки (головки)
                   ax=40,  # эти числа определяют положение текста относительно точки, и следовательно длину стрелки
                   ay=-50,
                   # подложка под подпись
                   bordercolor="#c2c2c2",  # цвет рамки
                   borderwidth=2,  # толщина рамки
                   borderpad=9,  # расстояние от текста до рамки
                   bgcolor="white",  # цвет подложки
                   opacity=0.8)  # непрозрачность

fig.update_traces(hovertemplate='<b>%{y}</b><br>Курс: %{facet_row}<br>Время выполнения: %{x}<extra></extra>')

# экспорт изображения графика в папку на компе
fig.write_image(r'D:\My\Programing\Graphs\Graphs_docs\{}.png'.format('cumulative_distribution_function'),
                width=1200,  # ширина изображения
                height=700,  # высота изображения
                scale=0.47)  # масштаб сохранения
