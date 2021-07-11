import plotly.express as px
import plotly.offline as offline
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.graph_objs as go


df = px.data.tips()

poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
x = np.sort(x)
y = np.sin(x) + 0.1 * rng.randn(50)
poly_model.fit(x[:, np.newaxis], y)
y_predict = poly_model.predict(x[:, np.newaxis])

df = pd.DataFrame({'x': x, 'y': y, 'y_predict': y_predict})


settings = dict(size=dict(small=dict(title=25, title_x=23, title_y=23, tick_x=18, tick_y=18),
                          medium=dict(title=20, x=20, y=20),
                          large=dict(title=20, x=20, y=20)))


def create_graph(dataframe, graph_settings):
    fig = go.Figure()

    # fig = px.scatter(dataframe, x="x", y="y")
    fig.add_trace(go.Scatter(x=dataframe.x, y=dataframe.y, mode='markers',
                             marker=dict(size=13)))

    fig.add_trace(go.Scatter(x=dataframe.x, y=dataframe.y_predict,
                             line=dict(
                                       width=6)))

    fig.add_trace(go.Scatter(
                  mode="markers+text",
                  x=[4, 8],
                  y=[0.5, -0.5],
                  text=["Point A", "Point B"]))

    # fig.add_trace(go.Scatter(x=dataframe.x, y=dataframe.y_predict, mode='markers',
    #                          marker_size=20,
    #                          marker_color="lightskyblue",
    #                          marker_line_color="midnightblue",
    #                          marker_line_width=2,
    #                          marker_symbol="x-dot"))

    fig.update_layout(title=dict(text='<b>Полиномиальная аппроксимация нелинейной обучающей последовательности</b>',
                                 font=dict(size=graph_settings['title'],
                                           family='Pt Sans',
                                           color='#5c5c5c')),
                      paper_bgcolor='#ffffff',
                      plot_bgcolor='#ffffff',
                      font_color='#888888',
                      font_family='#888888',
                      legend=dict(bordercolor='#e8e8e8',
                                  borderwidth=2),
                      margin=dict(l=100, r=100, t=90, b=80),
                      colorway=px.colors.qualitative.D3)

    fig.update_xaxes(title=dict(text='Подпись оси Х',
                                font=dict(size=graph_settings['title_x'],
                                          family='Pt Sans',
                                          color='#888888')),
                     tickfont=dict(family='Arial',
                                   color='#888888',
                                   size=graph_settings['tick_x']),
                     gridcolor='#d9d9d9',
                     zerolinewidth=3, zerolinecolor='#858585')

    fig.update_yaxes(title=dict(text='Подпись оси У',
                                font=dict(size=graph_settings['title_y'],
                                          family='Pt Sans',
                                          color='#888888')),
                     tickfont=dict(family='Arial',
                                   color='#888888',
                                   size=graph_settings['tick_y']),
                     gridcolor='#d9d9d9',
                     zerolinewidth=3, zerolinecolor='#858585')

    return fig


def export_image_graph():
    graph_object = create_graph(df, settings['size']['small'])
    return graph_object.write_image(r"C:\Users\Ilya\Desktop\image.png", width=1367, height=617, scale=0.47)


export_image_graph()
