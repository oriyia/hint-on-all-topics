import plotly.express as px
import plotly.offline as offline
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


theme_color = ['#4189c3', '#41c3a9', '#1ba672', '#6b737d', '#ffad38', '#ed5e73', '#c96dd0', '#4db2ff', '#825ec2']

docs_theme = dict(
    layout=go.Layout(colorway=theme_color,
                     title_font=dict(family="Helvetica", size=28, color='#5c5c5c'),
                     legend=dict(bordercolor='#e8e8e8', borderwidth=2,
                                 font=dict(family='Helvetica', size=22, color='#5c5c5c'),
                                 orientation='h',
                                 # x=0.7,
                                 y=-0.2),
                     paper_bgcolor='white',
                     plot_bgcolor='white',
                     xaxis=dict(gridcolor='#dbdbdb',
                                gridwidth=2,
                                zerolinecolor='#b8b8b8',
                                zerolinewidth=4,
                                title=dict(font_size=23,
                                           font_color='#5c5c5c',
                                           standoff=15,
                                           font_family='Helvetica'),
                                tickfont=dict(family='Helvetica',
                                              size=20,
                                              color='#5c5c5c'),
                                showline=True, linewidth=1, linecolor='#b8b8b8', mirror=True),
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
                     width=1367, height=617,
                     )
)


def plotting_points_graph(fig_object, x_coordinates, y_coordinates):
    """Функция построения проекций точек на оси графика"""
    for x_i, y_j in zip(x_coordinates, y_coordinates):
        # построение вертикальной линии
        fig_object.add_shape(
            type='line',
            line_dash='dash',
            x0=x_i, x1=x_i, y0=y_j, y1=0,
            line=dict(width=4, color='#c4c4c4'),
            layer='below',
        )
        # построение горизонтальной линии
        fig_object.add_shape(
            type='line',
            line_dash='dash',
            x0=x_i, x1=0, y0=y_j, y1=y_j,
            line=dict(width=4, color='#c4c4c4'),
            layer='below',
        )


def export_image_graph_png(fig_object, name='graph_image', export_width=None, export_height=None, export_scale=0.49):
    """Функция экспорта графика в изображение на компьютер, формат .png"""
    fig_object.write_image(
        r"D:\My\Programing\Graphs\Graphs_docs\{}.png".format(name),
        width=export_width,
        height=export_height,
        scale=export_scale,
    )


def sigmoid_graph():
    """Логистическая функция (сигмоида)"""

    y_axes = np.linspace(0.00001, 0.9999999, 100)
    x_axes = np.log(y_axes / (1 - y_axes))

    fig = go.Figure(go.Scatter(
        x=x_axes, y=y_axes,
        line=dict(width=8),
    ))

    fig.add_hline(
        y=1,
        line_dash='dash',
        line=dict(width=5, color='#8a8a8a'),
        layer='below',
    )

    fig.add_hline(
        y=0,
        line_dash='dash',
        line=dict(width=5, color='#8a8a8a'),
        layer='below',
    )

    fig.update_layout(
        title=dict(text='<b>Логистическая функция (сигмоида)</b>'),
        template=docs_theme,
    )

    fig.update_xaxes(
        range=[-5, 5],
        title=dict(text='y'),
    )

    fig.update_yaxes(
        title=dict(text='p'),
    )

    export_image_graph_png(fig, 'sigmoid_graph')


def logit_graph():
    """Типовой график logit(p) для диапазона [0,1] и основание e для логарифмирования"""

    x_axes = np.linspace(0.00001, 0.9999999, 100)
    y_axes = np.log(x_axes / (1 - x_axes))

    fig = go.Figure(go.Scatter(
        x=x_axes, y=y_axes,
        line=dict(width=8),
    ))

    fig.update_layout(
        title=dict(text='<b>Типовой график logit(p) для диапазона [0,1] и основание e для логарифмирования</b>'),
        template=docs_theme,
    )

    fig.update_xaxes(
        title=dict(text='$\Large{p}$')
    )

    fig.update_yaxes(
        range=[-5, 5],
        title=dict(text='$\Large{logit(p)}$')
    )

    export_image_graph_png(fig, 'logit_graph', 1200, 750)


def log_graph_other_parameter():
    """Логистические функции с разными параметрами"""

    x_axes = np.linspace(-70, 70, 140)
    y_axes1 = 1 / (1 + 2.7 ** (-(10 + 0.3 * x_axes)))
    y_axes2 = 1 / (1 + 2.7 ** (-(-2 + 0.8 * x_axes)))

    fig = go.Figure(go.Scatter(
        x=x_axes, y=y_axes1,
        line=dict(width=8),
        name=r'$\Large{\beta_{0}=10, \beta=0.3}$'),
    )

    fig.add_trace(go.Scatter(
        x=x_axes, y=y_axes2,
        line=dict(width=8),
        name=r'$\Large{\beta_{0}= -2, \beta=0.8}$'),
    )

    fig.add_hline(
        y=1,
        line_dash='dash',
        line=dict(color='#8a8a8a', width=5),
        layer="below",
    )

    fig.add_hline(
        y=0,
        line_dash='dash',
        line=dict(color='#8a8a8a', width=5),
        layer="below",
    )

    fig.update_layout(
        title=dict(text='<b>Логистические функции с разными параметрами</b>'),
        template=docs_theme,
    )

    fig.update_xaxes(
        title=dict(text='$\Large{y}$'),
    )

    fig.update_yaxes(
        title=dict(text='$\Large{p}$'),
    )

    export_image_graph_png(fig, 'log_graph_other_parameter')


def cumulative_distribution_function():
    """Функция распределения для ДСВ"""

    y_axes = [0, 27 / 64, 0.7, 0.9, 1]
    x_axes = [0, 1, 2, 3, 4]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_axes[:-1], y=y_axes[1:],
        mode='markers',
        marker=dict(size=17),
    ))

    fig.add_shape(
        type='line',
        x0=-10, x1=0, y0=0, y1=0,
        line=dict(width=8, color=theme_color[0]),
    )

    fig.add_shape(
        type='line',
        x0=0, x1=1, y0=y_axes[1], y1=y_axes[1],
        line=dict(width=8, color=theme_color[0]),
    )

    fig.add_shape(
        type='line',
        x0=1, x1=2, y0=y_axes[2], y1=y_axes[2],
        line=dict(width=8, color=theme_color[0]),
    )

    fig.add_shape(
        type='line',
        x0=2, x1=3, y0=y_axes[3], y1=y_axes[3],
        line=dict(width=8, color=theme_color[0]),
    )

    fig.add_shape(
        type='line',
        x0=3, x1=10, y0=y_axes[4], y1=y_axes[4],
        line=dict(width=8, color=theme_color[0]),
    )

    fig.update_layout(
        title=dict(text='<b>Функция распределения для ДСВ</b>'),
        template=docs_theme,
    )

    fig.update_xaxes(
        range=[-1, 4],
        title=dict(text='$\Large{x}$'),
    )

    fig.update_yaxes(
        range=[-0.1, 1.1],
        title=dict(
            text='$\Large{F_{X}}$',  # размер \large{}, \Large{}, \huge{}, \Huge{}
            font_size=30),
    )

    export_image_graph_png(fig, 'cumulative_distribution_function', export_width=700)


def distribution_function_properties():
    """Функция распределения и ее свойства"""

    y_axes = np.linspace(0.00001, 0.9999999, 100)
    x_axes = np.log(y_axes / (1 - y_axes)) + 3

    # построение первой части графика
    fig = go.Figure(go.Scatter(
        x=x_axes[:30], y=y_axes[:30],
        line=dict(width=8),
    ))

    # построение 2 части
    fig.add_shape(
        x0=x_axes[30], x1=x_axes[60], y0=0.5, y1=0.5,
        line=dict(width=8, color=theme_color[0]),
    )

    # построение 3 части
    fig.add_trace(go.Scatter(
        x=x_axes[60:], y=y_axes[60:],
        line=dict(width=8, color=theme_color[0]),
    ))

    # построение точек
    fig.add_trace(go.Scatter(
        x=[x_axes[60], x_axes[30]], y=[y_axes[60], 0.5],
        mode='markers',
        marker=dict(size=17, color=theme_color[0]),
    ))

    # построение пользовательской оси у
    fig.add_trace(go.Scatter(
        mode='markers+text',
        x=[0, 0, 0, 0, 0], y=[0, y_axes[30], y_axes[60], 1, 0.5],
        text=[str(round(i, 2)) for i in [0, y_axes[30], y_axes[60], 1, 0.5]],
        textposition='top left',  # (top, middle, bottom) (left, center, right)
        textfont=dict(
            family='Helvetica',
            size=20,
            color='#5c5c5c',
        ),
        marker=dict(
            size=14,
            line=dict(width=3),
            color='#b8b8b8',
            symbol='line-ne-open',
        )
    ))

    # подпись пользовательской оси
    fig.add_annotation(
        x=-0.8, y=0.5,
        text=r'$\huge{F_{X}(x)}$',
        showarrow=False,
        font=dict(
            family='Helvetica',
            size=14,
            color='#5c5c5c'),
        textangle=-90,
    )

    plotting_points_graph(fig, [x_axes[60], x_axes[30], x_axes[30]], [y_axes[60], y_axes[30], 0.5])

    fig.add_hline(
        y=1,
        line_dash='dash',
        line=dict(width=5, color='#c4c4c4'),
        layer='below',
    )

    fig.add_hline(
        y=0,
        line_dash='dash',
        line=dict(width=5, color='#c4c4c4'),
        layer='below',
    )

    fig.update_layout(
        title=dict(text='<b>Функция распределения и ее свойства</b>'),
        template=docs_theme,
        showlegend=False,
    )

    fig.update_xaxes(
        range=[-2, 7],
        title=dict(text='$\Large{x}$'),
    )

    fig.update_yaxes(
        title=dict(text='$\Large{F_{X}}$'),
        range=[-0.1, 1.1],
        visible=False,
    )

    export_image_graph_png(fig, 'distribution_function_properties', 1000)


def distribution_function_crv():
    """Функция распределения для НСВ"""

    x_array = np.linspace(-7, 7, 100)
    y_array = 1 / (1 + 2.7 ** (-x_array))
    x_array += 2

    fig = go.Figure(go.Scatter(
        x=x_array, y=y_array,
        line=dict(width=8),
    ))

    fig.add_trace(go.Scatter(
        x=[x_array[40], x_array[48], x_array[55]],
        y=[y_array[40], y_array[48], y_array[55]],
        mode='markers',
        marker=dict(size=20, color=theme_color[0]),
    ))

    plotting_points_graph(fig, [x_array[40], x_array[48], x_array[55]], [y_array[40], y_array[48], y_array[55]])

    # добавление пользовательских осей координат
    # ось у
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1.1],
        mode='lines',
        line=dict(width=4, color='#b8b8b8'),
    ))

    # ось х
    fig.add_trace(go.Scatter(
        x=[-5, 9], y=[0, 0],
        mode='lines',
        line=dict(width=4, color='#b8b8b8')
    ))

    # создание тиков и подписей ОУ
    fig.add_trace(go.Scatter(
        x=[0, 0, 0, 0, 0],
        y=[0, y_array[40], y_array[48], y_array[55], 1],
        mode='markers+text',
        text=['0', r'$\Large{F_{X}(x - \xi)}$', r'$\Large{F_{X}(x - \frac{\xi}{2})}$', r'$\Large{F_{X}(x)}$', '1'],
        textposition='bottom left',
        textfont=dict(
            family='Helvetica',
            size=20,
            color='#5c5c5c'
        ),
        marker=dict(
            size=14,
            color='#b8b8b8',
            line=dict(width=3),
            symbol='line-ne-open',
        )
    ))

    # создание тиков и подписей для ОХ
    fig.add_trace(go.Scatter(
        x=[x_array[40], x_array[48], x_array[55]],
        y=[0, 0, 0],
        mode='markers+text',
        text=[r'$\Large{(x - \xi)}$', r'$\Large{(x - \frac{\xi}{2})}$', r'$\Large{(x)}$'],
        textposition='bottom center',
        textfont=dict(
            family='Helvetica',
            size=20,
            color='#5c5c5c'
        ),
        marker=dict(
            size=14,
            color='#b8b8b8',
            line=dict(width=3),
            symbol='line-ne-open',
        )
    ))

    # создание стрелок осей координат с подписями
    fig.add_trace(go.Scatter(
        x=[0, 9], y=[1.1, 0],
        mode='markers+text',
        text=[r'$\Large{F_{X}(x)}$', r'$\Large{x}$'],
        textposition=['middle left', 'bottom center'],
        textfont=dict(family='Helvetica',
                      size=20,
                      color='#5c5c5c'),
        marker=dict(size=14,
                    color='#b8b8b8',
                    symbol=['triangle-up', 'triangle-right'])
    ))

    fig.add_hline(
        y=1,
        line_dash='dash',
        line=dict(color='#c4c4c4',
                  width=4),
        layer='below',
    )
    fig.add_hline(
        y=-0.15,
        line_dash='dash',
        line=dict(color='#c4c4c4',
                  width=0),
        layer='below',
    )

    fig.update_xaxes(
        visible=False,
    )

    fig.update_yaxes(
        visible=False,
    )

    fig.update_layout(
        title=dict(text='<b>Функция распределения для НСВ</b>'),
        template=docs_theme,
        showlegend=False,
    )

    export_image_graph_png(fig, 'distribution_function_crv', 1200)


def probability_density_function():
    """График плотности вероятности"""

    from scipy.stats import gamma
    lambda_g = 2
    k = 1
    x_axes = np.linspace(0, 6, 200)
    gamma_distribution = gamma(lambda_g, 0, k)
    pdf = gamma_distribution.pdf(x_axes)

    fig = go.Figure(go.Scatter(
        x=x_axes, y=pdf,
        mode='lines',
        line=dict(width=8)
    ))

    # подпись выделенной интегрированной области
    fig.add_annotation(
        x=2, y=0.15,
        text=r"$\Huge{P(a\leq X\leq b)=\int_a^b f_{X}(\xi)d\xi}$",
        ax=400, ay=-100,
        arrowwidth=3,
        arrowsize=1,
        arrowhead=5,
        arrowcolor='#757575',
    )

    # выделенная область интегрирования
    fig.add_trace(go.Scatter(
        x=x_axes[40:80],
        y=pdf[40:80],
        line=dict(width=8, color=theme_color[0]),
        fill='tozeroy',
    ))

    # тики по оси у выделенной области
    fig.add_trace(go.Scatter(
        x=[x_axes[40], x_axes[79]], y=[0, 0],
        mode='markers+text',
        text=[r'$\huge{a}$', r'$\huge{b}$'],
        textposition='bottom center',
        marker=dict(
            size=20,
            line=dict(width=4),
            color='#b8b8b8',
            symbol='line-ns-open',
        ),
    ))

    fig.update_layout(
        title=dict(text='<b>График плотности вероятности</b>'),
        template=docs_theme,
        showlegend=False,
    )

    fig.update_xaxes(
        title=dict(text=r'$\Large{x}$')
    )

    fig.update_yaxes(
        title=dict(text=r'$\Large{f_{X}(x)}$')
    )

    export_image_graph_png(fig, 'probability_density_function', 1100, 530)


def probability_density_function2():

    from scipy.stats import gamma

    lambda_g = 4
    k = 1
    x_axes = np.linspace(-1 , 9, 200)
    gamma_distribution = gamma(lambda_g, 0, k)
    y_axes = gamma_distribution.pdf(x_axes)

    fig = go.Figure(go.Scatter(
        x=x_axes, y=y_axes,
        line=dict(width=8)
    ))

    # участки интегрирования
    fig.add_trace(go.Scatter(
        x=x_axes[40:60], y=y_axes[40:60],
        line=dict(color=theme_color[0]),
        fill='tozeroy',
    ))

    fig.add_trace(go.Scatter(
        x=x_axes[80:90], y=y_axes[80:90],
        line=dict(color=theme_color[0]),
        fill='tozeroy',
    ))

    # текст со стрелкой
    fig.add_annotation(
        x=1.5, y=0.08,
        text=r'$\Huge{P(X \in A)= \int_a^b f_{X}(x)dx + \int_c^d f_{X}(x)dx}$',
        ax=560, ay=-100,
        arrowhead=5,
        arrowwidth=2,
        arrowcolor='#757575',
        bgcolor='white',
        borderpad=4,
        borderwidth=2,
        bordercolor='#c2c2c2',
    )

    # вторая стрелка
    fig.add_annotation(
        x=3.2, y=0.08,
        text='',
        ax=230, ay=-74,
        arrowhead=5,
        arrowcolor='#757575',
    )

    fig.add_trace(go.Scatter(
        x=[x_axes[40], x_axes[59], x_axes[80], x_axes[89]], y=[0, 0, 0, 0],
        mode='markers+text',
        text=[r'$\huge{a}$', r'$\huge{b}$', r'$\huge{c}$', r'$\huge{d}$',],
        textposition='bottom center',
        marker=dict(
            size=20,
            color='#b8b8b8',
            line=dict(width=4),
            symbol='line-ns-open',
        )
    ))

    fig.update_xaxes(
        title=dict(text=r'$\Large{x}$'),
        range=[0, 9]
    )

    fig.update_yaxes(
        title=dict(text=r'$\Large{f_{X}(x)}$'),
        range=[-0.05, 0.25],
    )

    fig.update_layout(
        title=dict(text='<b>Плотность вероятности для нескольких участков</b>'),
        template=docs_theme,
        showlegend=False,
    )

    export_image_graph_png(fig, 'probability_density_function2', 1100, 530)


def stochastic_dominance():

    x_axes = np.linspace(-9, 9, 100)
    y_axes = 1 / (1 + 2.7 ** (-x_axes))
    x_axes1 = x_axes + 3
    x_axes2 = x_axes + 6

    fig = go.Figure(go.Scatter(
        x=x_axes1, y=y_axes,
        mode='lines',
        line=dict(width=8),
    ))

    fig.add_trace(go.Scatter(
        x=x_axes2, y=y_axes,
        mode='lines',
        line=dict(width=8)
    ))

    # подписи графиков со стрелками
    fig.add_annotation(
        x=3, y=0.58,
        text=r'$\Huge{F_{Y}(\xi)}$',
        arrowhead=5,
        arrowcolor='#757575',
        ax=-50, ay=-59,
    )

    fig.add_annotation(
        x=6, y=0.4,
        text=r'$\Huge{F_{X  }(\xi)}$',
        arrowhead=5,
        arrowcolor='#757575',
        ax=50, ay=59,
    )

    fig.update_layout(
        template=docs_theme,
        showlegend=False,
    )

    fig.update_xaxes(
        title=dict(text=r'$\Large{\xi}$'),
        range=[-3, 11],
    )

    export_image_graph_png(fig, 'stochastic_dominance', 1100)


def likelihood_sample():
    """График правдоподобности выборки"""

    # задание распределения
    from scipy.stats import norm
    x_axes1 = np.linspace(-14, 26, 100)
    # Mean = 0, SD = 2.
    pdf1 = norm(6, 2).pdf(x_axes1)
    marker_x = [5.2, 7, 8.2]
    markers_pdf1 = norm(6, 2).pdf(marker_x)
    pdf2 = norm(8, 2).pdf(x_axes1)
    markers_pdf2 = norm(8, 2).pdf(marker_x)

    # для дальнейшего легкого выбора точек
    # df_axes = pd.DataFrame({'x_axes1': x_axes1, 'x_axes2': x_axes2, 'pdf': pdf})

    fig = go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.02,
    )

    fig.add_trace(go.Scatter(
        x=x_axes1, y=pdf1,
        mode='lines',
        line=dict(width=8),
    ), 1, 1)

    fig.add_trace(go.Scatter(
        x=x_axes1, y=pdf2,
        mode='lines',
        line=dict(width=8),
    ), 2, 1)

    # добавление пользовательских осей координат
    fig.add_shape(
        type='line',
        x0=0, x1=0, y0=-0.01, y1=0.2,
        line=dict(width=4, color='#b8b8b8'),
        row=2, col=1,
        layer='below',
    )

    fig.add_shape(
        type='line',
        x0=0, x1=0, y0=-0.01, y1=0.2,
        line=dict(width=4, color='#b8b8b8'),
        row=1, col=1,
        layer='below',
    )

    fig.add_shape(
        type='line',
        x0=-2, x1=16, y0=0, y1=0,
        line=dict(width=4, color='#b8b8b8'),
        row=2, col=1,
        layer='below',
    )

    fig.add_shape(
        type='line',
        x0=-2, x1=16, y0=0, y1=0,
        line=dict(width=4, color='#b8b8b8'),
        row=1, col=1,
        layer='below',
    )

    # добавление стрелки для пользовательской оси координат,
    fig.add_trace(go.Scatter(
        x=[0], y=[0.2],
        mode='markers',
        marker=dict(
            size=16,
            line=dict(width=0),
            color='#b8b8b8',
            symbol='triangle-up',
        )
    ), 1, 1)

    fig.add_trace(go.Scatter(
        x=[0], y=[0.2],
        mode='markers+text',
        marker=dict(
            size=16,
            line=dict(width=0),
            color='#b8b8b8',
            symbol='triangle-up',
        )
    ), 2, 1)

    # добавление подписей для графиков
    fig.add_annotation(
        x=2, y=0.15,
        text=r'$\huge{p_{0}(x|H_{0})}$',
        showarrow=False,
        font=dict(
            family='Helvetica',
            size=23,
            color='#5c5c5c'
        ),
        ax=0, ay=0,
        bordercolor="#c2c2c2",  # цвет рамки
        borderwidth=2,  # толщина рамки
        borderpad=9,  # расстояние от текста до рамки
        bgcolor="white",  # цвет подложки
        row=1, col=1,
    )

    fig.add_annotation(
        x=2, y=0.15,
        text=r'$\huge{p_{1}(x|H_{1})}$',
        showarrow=False,
        font=dict(
            family='Helvetica',
            size=23,
            color='#5c5c5c'
        ),
        ax=0, ay=0,
        bordercolor="#c2c2c2",  # цвет рамки
        borderwidth=2,  # толщина рамки
        borderpad=9,  # расстояние от текста до рамки
        bgcolor="white",  # цвет подложки
        row=2, col=1,
    )

    # добавление рассматриваемых точек
    fig.add_trace(go.Scatter(
        x=marker_x,
        y=markers_pdf1,
        mode='markers',
        marker=dict(
            size=16,
            line=dict(width=5, color=theme_color[0]),
            color='white',
        )
    ), 1, 1)
    
    fig.add_trace(go.Scatter(
        x=marker_x,
        y=markers_pdf2,
        mode='markers',
        marker=dict(
            size=16,
            line=dict(width=5, color=theme_color[1]),
            color='white',
        )
    ), 2, 1)

    # добавление горизонтальных пунктирных линий
    for x, y in zip(marker_x, markers_pdf1):
        fig.add_shape(
            x0=x, y0=y, x1=x, y1=0,
            line_dash='dash',
            line=dict(width=4, color='#c4c4c4'),
            layer='below'
        )

    for x, y in zip(marker_x, [0.21, 0.21, 0.21,]):
        fig.add_shape(
            x0=x, y0=y, x1=x, y1=0,
            line_dash='dash',
            line=dict(width=4, color='#c4c4c4'),
            layer='below',
            row=2, col=1,
        )

    # добавление подписи тиков
    fig.add_trace(go.Scatter(
        x=marker_x, y=[0, 0, 0],
        mode='markers+text',
        text=[r'$\huge{x_{i}}$', r'$\huge{x_{i+1}}$', r'$\huge  {x_{i+2}}$'],
        textposition='bottom center',
        marker=dict(
            line=dict(width=4),
            size=13,
            color='#b8b8b8',
            symbol='line-ns-open',
        ),
        textfont=dict(
            size=23,
            color='#5c5c5c',
        ),
    ))

    fig.update_layout(
        title=dict(text='<b>Правдоподобность выборки</b>'),
        template=docs_theme,
        showlegend=False,
    )

    fig.update_xaxes(
        range=[-1, 18],
        showticklabels=False,
        # visible=False,
    )

    fig.update_yaxes(
        # visible=False,
        showticklabels=False,
        range=[-0.03, 0.21],
    )

    export_image_graph_png(fig, 'likelihood_sample', export_height=800)


def example():
    fig = go.Figure()

    x = [0, 1, 2]
    y1 = [0, 1, 2]

    fig.add_trace(go.Scatter(
        x=x, y=y1,
        mode='markers',
        marker=dict(
            size=35,
            line=dict(width=5, color='DarkSlateGrey'),
            color='LightSkyBlue',
            opacity=0.8,
            symbol='square-dot'),
        opacity=0.9,
    ))

    fig.update_layout(
        width=1360,
        height=710,
        template=docs_theme,
        hovermode='closest',
    )

    fig.update_xaxes(
        title=dict(
            text='Название оси Х dd',
            font=dict(
                family='Helvetica',
                size=25,
                color='grey',
            )),
        showspikes=True,
        spikemode='across',
        spikecolor='black',
        spikethickness=10,
        showline=True,
        linewidth=10,
    )

    fig.write_image(
        r"D:\My\Programing\Graphs\Graphs_docs\{}.png".format('example'),
        scale=0.47,
    )


# example()
# logit_graph()
# log_graph_other_parameter()
# sigmoid_graph()
# cumulative_distribution_function()
# distribution_function_properties()
# distribution_function_crv()
# probability_density_function(
# probability_density_function2()
# stochastic_dominance()
likelihood_sample()
