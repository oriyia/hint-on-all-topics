import plotly.express as px
import plotly.offline as offline
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.graph_objs as go
import enum

df = px.data.tips()

poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
x = np.sort(x)
y = np.sin(x) + 0.1 * rng.randn(50)
poly_model.fit(x[:, np.newaxis], y)
y_predict = poly_model.predict(x[:, np.newaxis])

df1 = pd.DataFrame({'x': x, 'y': y, 'y_predict': y_predict})


import plotly.io as pio


# docs_theme = go.layout.Template()
# docs_theme.layout.annotations = [
#
# ]
temp_color = ['#687386', '#586ada', '#0579b7', '#09815c', '#bb5505', '#c44b35', '#cd3d64', '#a44eb4', '#825ec2']
docs_theme = dict(
    layout=go.Layout(colorway=['#41c3a9', '#415bc3', '#6e41c3','#4189c3'],
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
                     width=1367, height=617))  # 1367 617


def create_graph(dataframe):
    fig = go.Figure()

    # fig = px.scatter(dataframe, x="x", y="y")
    fig.add_trace(go.Scatter(x=dataframe.x, y=dataframe.y, mode='markers',
                             marker_size=12))

    fig.add_trace(go.Scatter(x=dataframe.x, y=dataframe.y_predict,
                             line=dict(width=7)))

    fig.add_trace(go.Scatter(mode="markers+text",
                             x=[4, 8],
                             y=[0.5, -0.5],
                             text=["Point A", "Point B"]))

    # fig.add_trace(go.Scatter(x=dataframe.x, y=dataframe.y_predict, mode='markers',
    #                          marker_size=20,
    #                          marker_color="lightskyblue",
    #                          marker_line_color="midnightblue",
    #                          marker_line_width=2,
    #                          marker_symbol="x-dot"))

    fig.update_layout(title=dict(text='<b>Полиномиальная аппроксимация нелинейной обучающей последовательности</b>'),
                      template=docs_theme)

    fig.update_xaxes(title=dict(text='Очень длинное название подписи оси Х'))

    fig.update_yaxes(title=dict(text='Длинное название подписи оси У'))

    # fig.update_layout(title=dict(text='<b>Полиномиальная аппроксимация нелинейной обучающей последовательности</b>',
    #                              font=dict(size=graph_settings['title'],
    #                                        family='Pt Sans',
    #                                        color='#5c5c5c')),
    #                   paper_bgcolor='#ffffff',
    #                   plot_bgcolor='#ffffff',
    #                   font_color='#888888',
    #                   font_family='#888888',
    #                   legend=dict(bordercolor='#e8e8e8',
    #                               borderwidth=2),
    #                   margin=dict(l=100, r=100, t=90, b=80),
    #                   colorway=px.colors.qualitative.D3)
    #
    # fig.update_xaxes(title=dict(text='Подпись оси Х',
    #                             font=dict(size=graph_settings['title_x'],
    #                                       family='Pt Sans',
    #                                       color='#888888')),
    #                  tickfont=dict(family='Arial',
    #                                color='#888888',
    #                                size=graph_settings['tick_x']),
    #                  gridcolor='#d9d9d9',
    #                  zerolinewidth=3, zerolinecolor='#858585')
    #
    # fig.update_yaxes(title=dict(text='Подпись оси У',
    #                             font=dict(size=graph_settings['title_y'],
    #                                       family='Pt Sans',
    #                                       color='#888888')),
    #                  tickfont=dict(family='Arial',
    #                                color='#888888',
    #                                size=graph_settings['tick_y']),
    #                  gridcolor='#d9d9d9',
    #                  zerolinewidth=3, zerolinecolor='#858585')
    # offline.plot(fig)
    return fig


def export_image_graph(dataframe, name):
    graph_object = create_graph(dataframe)
    return graph_object.write_image(r"C:\Users\Ilya\Desktop\{}.png".format(name), scale=0.47)


export_image_graph(df1, 'graph2')




x_axes = np.linspace(0.00001, 0.9999999, 100)
y_axes = np.log(x_axes / (1 - x_axes))

df2 = pd.DataFrame({'x': x_axes, 'y': y_axes})

fig = go.Figure(go.Scatter(x=x_axes, y=y_axes,
                           line=dict(width=7)))


# fig.add_hline(y=1, line_dash='dash', line=dict(color='#8cc2ff'))
# fig.add_hline(y=0, line_dash='dash', line=dict(color='#8cc2ff'))
fig.update_layout(title=dict(text='<b>Типовой график logit(p) для диапазона [0,1] '
                                  'и основание e для логарифмирования</b>'),
                  template=docs_theme)
fig.update_xaxes(title=dict(text='p'))
fig.update_yaxes(range=[-5, 5],
                 title=dict(text='logit(p)'))
# offline.plot(fig)


# fig.write_image(r"C:\Users\Ilya\Desktop\{}.png".format('logit'), scale=0.47)



x_axes = np.linspace(-70, 70, 140)
y_axes1 = 1 / (1 + 2.7 ** (-(10 + 0.3 * x_axes)))
y_axes2 = 1 / (1 + 2.7 ** (-(-2 + 0.8 * x_axes)))
y_axes3 = 1 / (1 + 2.7 ** (-(-3 + 0.8 * x_axes)))
y_axes4 = 1 / (1 + 2.7 ** (-(-2 + 0.8 * x_axes)))
y_axes5 = 1 / (1 + 2.7 ** (-(-2 + 0.8 * x_axes)))
y_axes6 = 1 / (1 + 2.7 ** (-(-2 + 0.8 * x_axes)))
y_axes7 = 1 / (1 + 2.7 ** (-(-2 + 0.8 * x_axes)))
y_axes8 = 1 / (1 + 2.7 ** (-(-2 + 0.8 * x_axes)))
y_axes9 = 1 / (1 + 2.7 ** (-(-2 + 0.8 * x_axes)))
y_axes10 = 1 / (1 + 2.7 ** (-(-2 + 0.8 * x_axes)))

df2 = pd.DataFrame({'x': x_axes, 'y1': y_axes1, 'y2': y_axes2})

fig = go.Figure(go.Scatter(x=x_axes, y=y_axes1,
                           line=dict(width=8),
                           name='beta0=10, beta=0.3'))

fig.add_trace(go.Scatter(x=x_axes, y=y_axes2,
                         line=dict(width=8),
                         name='beta0= -2, beta=0.8'))


fig.add_hline(y=1, line_dash='dash', line=dict(color='#8a8a8a', width=5), layer="below")
fig.add_hline(y=0, line_dash='dash', line=dict(color='#8a8a8a', width=5), layer="below")
fig.add_hline(y=0.9, line=dict(color='#6e41c3', width=7))
fig.add_hline(y=0.8, line=dict(color='#c341b8', width=7))
fig.add_hline(y=0.7, line=dict(color='#c34179', width=7))
fig.add_hline(y=0.6, line=dict(color='#c34141', width=7))
fig.add_hline(y=0.5, line=dict(color='#e39e44', width=7))
fig.add_hline(y=0.4, line=dict(color='#e3e344', width=7))
fig.add_hline(y=0.3, line=dict(color='#a3e344', width=7))
fig.add_hline(y=0.2, line=dict(color='#44e347', width=7))

fig.update_layout(title=dict(text='<b>Логистические функции с разными параметрами</b>'),
                  template=docs_theme)
fig.update_xaxes(title=dict(text='y'))
fig.update_yaxes(title=dict(text='p'))
# offline.plot(fig)


fig.write_image(r"C:\Users\Ilya\Desktop\{}.png".format('log_function_other_parameter'), scale=0.47)