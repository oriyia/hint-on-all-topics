# import plotly.plotly as py  # импорт для онлайна
import plotly.offline as offline  # импорт для оффлайна
from plotly.graph_objs import *  # импорт всех граф объекты
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


x_axes = [0, 1, 2]
y_axes = [0, 1, 2]


fig = go.Figure()  # создание поля для графиков
# распределение графиков по областям поля
fig = make_subplots(
    rows=3, cols=2,  # задание количества строк и столбцов
    subplot_titles=('Plot1', 'Plot2', 'Plot3', 'Plot4'),  # название для каждого графика
    specs=[[{'rowspan': 3}, {}], [None, {}], [None, {}]],  # расположение каждого графика на сетке
    vertical_spacing=0.02,  # расстояние между графиками по вертикали
    horizontal_spacing=0.03,  # расстояние между графиками по горизонтали
    column_widths=[2, 1],  # размерное отношение между графиками по строке
    row_heights=[3, 2, 1],  # размерное отношение между графиками по столбцу
)

# настройка оси для определенного графика поля
fig.update_yaxes(
    range=[-0.5, 5],
    zerolinecolor='LightPink',
    col=2,  # указываем столбец
)

fig.update_xaxes(
    range=[-0.5, 5],
    zerolinecolor='#000',
    col=2,
)

fig.add_trace(go.Scatter(x=x_axes, y=y_axes, name='f(x)=x<sup>2</sup>', mode='lines+markers'), 1, 2)
fig.add_trace(go.Scatter(x=x_axes, y=y_axes, name='f(x)=0.5x', mode='lines+markers',
                         marker=dict(color=x_axes, colorbar=dict(title='f(x)=x'),
                                     colorscale='Inferno', size=10*t(x))), 2, 2)

fig.add_trace(go.Scatter(
    x=x, y=x,
    name='g(x)=x',
    mode='markers',
    marker=dict(
        color='LightSkyBlue',
        size=10,
        line=dict(color='MediumPurple', width=2))
), 1, 2)

fig.add_trace(go.Scatter(x=x, y=h(x), name='h(x)=sin(x)'), 3, 2)

fig.add_trace(go.Scatter(x=x, y=g(x), name='g(x)=cos(x)'), 3, 2)

fig.add_trace(go.Scatter(
    x=x, y=j(x),
    visible='legendonly',  # неактивна функция, скрыта
    name='j(x)=tg(x)'
))

fig.add_trace(go.Scatter(
    x=x, y=j(x),
    name='j(x)=tg(x)'
),
    1, 1)

fig.update_layout(
    legend_orientation="h",  # внизу располагается легенда
    legend=dict(x=.5, xanchor="center"),  # выравнивание легенды (по центру)
    hovermode='x',  # при наведении на один график, данные отображаются и по второму (по иксу)
)

fig.update_xaxes(
    title='Ось Х графика 1',  # подписывание осей конкретного графика
    col=1, row=1
)

fig.update_xaxes(
    title='Ось Х графика 2',
    col=2, row=1
)

fig.update_yaxes(
    title='Ось Y графика 1',
    col=1, row=1
)

fig.update_yaxes(
    title='Ось Y графика 2',
    col=2, row=1
)

fig.update_traces(
    hoverinfo='all',
    hovertemplate='Аргумент: %{x}<br>Функция: %{y}'
)


# шаблон цветовой палитры
theme_color = ['#4189c3', '#41c3a9', '#1ba672', '#6b737d', '#ffad38', '#ed5e73', '#c96dd0', '#4db2ff', '#825ec2']

# настройки шаблона пользовательской темы
docs_theme = dict(
    layout=go.Layout(colorway=theme_color,  # используемая цветовая схема
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

# настройки графика для режима express
dataframe_object = pd.DataFrame()
fig = px.line(
    dataframe_object,
    x=module_statistics_group_count['mean'].values,
    y=module_statistics_group_count.index.get_level_values(1),
    facet_col=module_statistics_group_count.index.get_level_values(0),
    facet_col_wrap=1,
    labels=dict(x='Длительность', color='Курс', facet_col='Курс'),
    color=module_statistics_group_count.index.get_level_values(0),  #
    facet_row_spacing=0.025,  # расстояние между графиками по вертикали
)

# просто добавить фигуру c подписями точек
fig.add_trace(go.Scatter(
    x=x_axes, y=y_axes,
    mode='lines+text',  # lines+text - линия и подпись точек (их не видно) (markers+text,
    # lines+markers+text)
    text=["Text G", "Text H", "Text I"],  # надписи
    textposition="bottom center",  # положение текста (outside, inside,
    texttemplate='%{text:.2s}',  # формат текста
    textfont=dict(family='Helvetica',  # настройка текста подписей точек
                  size=18,
                  color='grey'),
    line=dict(width=8),
    name='beta0= -2, beta=0.8',
))

# добавление тиков для оси с подписями к ним
fig.add_trace(go.Scatter(
    mode='markers+text',
    x=[0, 0, 0, 0, 0], y=[0, y_axes[30], y_axes[60], 1, 0.5],
    text=[str(round(i, 2)) for i in [0, y_axes[30], y_axes[60], 1, 0.5]],
    textposition=['middle left', 'bottom center'],   # разные позиции для каждой точки
    textfont=dict(
        family='Helvetica',
        size=20,
        color='#5c5c5c',
    ),
    marker=dict(
        size=14,
        line=dict(width=3),
        color='#b8b8b8',
        symbol=['triangle-up', 'triangle-right'],  # свой символ для каждой точки
    ),
))

# маркеры (настройка стиля)
# https://plotly.com/python/marker-style/ - посмотреть остальные типы маркера
fig.add_trace(go.Scatter(
    x=x_axes, y=y_axes, mode='markers',
    marker=dict(size=35,  # размер маркера
                line=dict(width=5,  # толщина обводки
                          color='DarkSlateGrey'),  # цвет обводки
                color='LightSkyBlue',  # цвет маркера
                opacity=0.8,  # непрозрачность маркера
                symbol='square-dot'),  # тип маркера
    opacity=0.9,
))

# подпись точек на графике
fig.add_trace(go.Scatter(
    mode="markers+text",
    x=[4, 8],
    y=[0.5, -0.5],
    text=["Point A", "Point B"],
))

# горизонтальная линия
fig.add_hline(
    y=1,  # координата через которую проходит линия
    line_dash='dash',  # тип линии (dash - пунктирная, dot - точки
    line=dict(color='#c4c4c4',
              width=5),
    annotation_text='Подпись линии',
    annotation_position='bottom right',  # положение подписи
    annotation=dict(font=dict(family='Helvetica',
                              size=20,
                              color='grey')),
    layer="above",  # слой (below traces - на заднем плане, above traces - на переднем плане)
)

fig.add_shape(
    type='line',  # добавить линию
    x0=3, x1=10, y0=0, y1=2,
    line=dict(width=8, color='grey')
)

fig.update_layout(
    title=dict(
        text='<b>Название графика</b>',
        x=.5,
        xanchor="center",
    ),
    showlegend=False,  # не показывать легенду
    width=1200,  # ширина изображения, но не самого графика
    height=2200,  # высота изображения
    margin=dict(l=30, r=10, t=50, b=10),
    template='название темы',  # использование темы оформления
)

fig.update_xaxes(
    fixedrange=True,  # размер графика по всей подложке
    tickfont=dict(family='Helvetica',
                  size=15,  # размер тиков
                  color='grey'),
    showticklabels=True,  # показать все тики для (False - не показывать тики)
    ticksuffix=' дн.',  # суффикс у тиков оси
    tickprefix="$",  # приставка у тиков оси
    title=dict(text='$\Large{F_{X}}$',  # подпись оси x в Latex (\large{}, \Large{}, \huge{}, \Huge{})
               font=dict(family='Helvetica',  # шрифт
                         size=10,  # размер шрифта
                         color='grey'),  # цвет
               standoff=1),  # расстояние до графика
    nticks=6,  # количество тиков
    tickangle=45,  # угол наклона тиков
    tickformat='% H ~% M ~% S.% 2f',  # «2016-10-13 09: 15: 23.456» ---> «09 ~ 15 ~ 23.46»
    range=[3, 15],  # область показа значений на графике
    visible=False,  # скрыть оси у графика
    mirrir='ticks',
)

fig.update_yaxes(
    fixedrange=True,
    tickfont_size=9,
    side='right',  # сторона отображения тиков
    title=dict(
        text='Модули',  # подпись оси у
        font=dict(size=10)  # размер шрифта
    )
)

# добавить вертикальную форму
fig.add_vrect(
    x0=0.5, x1=2,  # начало и конец
    annotation_text="decline",  # надпись
    annotation_position="top left",  # позиция на форме
    annotation=dict(font=dict(family="Helvetica",  # шрифт надписи
                              size=20,  # размер
                              color='grey')),
    fillcolor="green",  # цвет формы (если убрать данный параметр, то будет прозрачная область)
    opacity=0.25,  # прозрачность
    line_width=0,
    line_color='grey',  # толщина обводки
)

# добавить овальную форму (область точек по максимуму и минимуму)
fig.add_shape(
    type="circle",  # тип фигуры
    xref="x", yref="y",
    x0=min(x_axes), y0=min(y_axes),  # овал строится по 2-м точка прямоугольника, куда вписывается овал
    x1=max(x_axes), y1=max(y_axes),  # 1 точка - левый нижний угол, 2 точка - правый верхний
    opacity=0.2,
    fillcolor="blue",
    line_color="blue"
)

# Добавить просто текст (координаты середины текста) 2 надписи в данной случае
fig.add_trace(go.Scatter(
    x=[2, 6], y=[1, 1],
    text=["Line positioned relative to the plot", "Line positioned relative to the axes"],
    mode="text"
))

# добавить произвольную форму с заливкой (можно указать область графика)
fig.add_trace(go.Scatter(
    x=[0, 1, 2, 0], y=[0.2, 0.6, 0.2, 0.2],
    line=dict(widht=8, color=theme_color[0]),  # определяя цвет линии, автоматически определяем цвет области
    fill="tozeroy"
))

# подпись с подложкой
fig.add_annotation(
    x=0, y=1,  # координаты точки для подпись
    text="Text annotation with arrow",  # сам текст подписи
    textangle=45,  # угол наклона текста
    showarrow=True,  # со стрелкой
    font=dict(family='Helvetica',  # шрифт подписи
              size=25,
              color='grey'),
    arrowwidth=3,  # толщина стрелки
    arrowcolor='#757575',  # цвет стрелки
    arrowsize=1,  # размер самой стрелки (головки)
    ax=40,  # определяют положение текста относительно точки, и следовательно длину стрелки
    ay=-50,
    standoff=10,  # отступ стрелки от указанной точки
    arrowhead=1,  # тип стрелки (1-5 разные виды стрелок, 6 - точка, 7 - квадратная точка)
    # подложка под подпись
    bordercolor="#c2c2c2",  # цвет рамки
    borderwidth=2,  # толщина рамки
    borderpad=9,  # расстояние от текста до рамки
    bgcolor="white",  # цвет подложки
    opacity=0.8,  # непрозрачность
)

fig.update_traces(
    hovertemplate='<b>%{y}</b><br>Курс: %{facet_row}<br>Время выполнения: %{x}<extra></extra>'
)

# экспорт изображения графика в папку на компе
fig.write_image(
    r'D:\My\Programing\Graphs\Graphs_docs\{}.png'.format('graph_name'),
    width=1200,  # ширина изображения
    height=700,  # высота изображения
    scale=0.47,  # масштаб сохранения
)


# построение проекций точек на оси графика
def plotting_points_graph(fig_object, x_coordinates, y_coordinates):

    for x_i, y_j in zip(x_coordinates, y_coordinates):
        # построение первой линии
        fig_object.add_shape(
            type='line',
            line_dash='dash',
            x0=x_i, x1=x_i, y0=y_j, y1=0,
            line=dict(width=4, color='#c4c4c4'),
            layer='below',
        )
        # построение второй линии
        fig_object.add_shape(
            type='line',
            line_dash='dash',
            x0=x_i, x1=0, y0=y_j, y1=y_j,
            line=dict(width=4, color='#c4c4c4'),
            layer='below',
        )


def create_histogram_graph():
    """Гистограмма"""
    fig = px.histogram(
        x_axes,
        nbins=100,  # количество разбиений
        histnorm='probability density',  # тип нормализации гистограммы (здесь плотность вероятности)
        cumulative=True,  # кумулятивная функция
        opacity=0.8,  # насыщенность графика
        marginal='box',  # дополнительное частотное распределение возле графика (rug - насечки)
    )
    return fig


def create_distplot_graph():
    """График распределения"""
    import plotly.figure_factory as ff
    fig = ff.create_distplot(
        x_axes,
        ['name_group1', 'name_group2'],
        colors='название столбца',  # изменение цвета графика
        curve_type='normal',  # тип изгиба кривой (kde=true)
        show_hist=False,  # не показывать гистограмму
        show_curve=False,  # не показывать кривую
        show_rug=False,
        bin_size=[.1, .6])  # количество разбиений, для каждой группы отдельно
    return fig
