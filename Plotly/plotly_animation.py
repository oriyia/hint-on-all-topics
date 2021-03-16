import plotly.offline as offline  # импорт для оффлайна
offline.init_notebook_mode()  # для использования в ноутбуке
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

x = np.arange(0, 5, 0.1)
def f(x):
    return x**2
def h(x):
    return np.sin(x)
def k(x):
    return np.cos(x)


num_steps = len(x)
fig = go.Figure(data=[go.Scatter(x=[x[0]], y=[h(x)[0]], mode='lines+markers', name='h(x)=sin(x)',
                                 marker=dict(color=[f(x[0])], colorbar=dict(yanchor='top', y=0.8, title="f(x)=x<sup>2</sup>"), colorscale='Inferno', size=[50*abs(h(x[0]))])),
                      go.Scatter(x=[x[0]], y=[k(x)[0]], mode='lines+markers', name='k(x)=cos(x)',
                                 marker=dict(color=[f(x[0])], colorscale='Inferno', size=[50*abs(k(x[0]))]))])

frames=[]
for i in range(0, len(x)):
    frames.append(go.Frame(name=str(i),
                           data=[go.Scatter(x=x[:i+1], y=h(x[:i+1]), mode='lines+markers', name='h(x)=sin(x)',
                                            marker=dict(color=f(x[:i+1]), colorscale='Inferno', size=50*abs(h(x[:i+1])))),
                                 go.Scatter(x=x[:i+1], y=k(x[:i+1]), mode='lines+markers', name='k(x)=cos(x)',
                                            marker=dict(color=f(x[:i+1]), colorscale='Inferno', size=50*abs(k(x[:i+1]))))]))

steps = []
for i in range(num_steps):
    step = dict(
        label = str(i),
        method = "animate",
        args = [[str(i)]]
    )
    steps.append(step)

sliders = [dict(
    steps = steps,
)]

fig.update_layout(updatemenus=[dict(direction="left",
                                    x=0.5,
                                    xanchor="center",
                                    y=0,
                                    showactive=False,
                                    type="buttons",
                                    buttons=[dict(label="►", method="animate", args=[None, {"fromcurrent": True}]),
                                             dict(label="❚❚", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                                               "mode": "immediate",
                                                                                               "transition": {"duration": 0}}])])],
                  )


fig.layout.sliders = sliders
fig.frames = frames

offline.plot(fig)
