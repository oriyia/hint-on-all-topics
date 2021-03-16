from scipy.stats import chi2_contingency
import numpy as np
import plotly.express as px
import plotly.offline as offline

table = [[1, 2, 3], [1, 2, 3]]
# задание нормального распределения
np.random.seed(0)  # - для какой-то там нормализации
x_rand = np.random.normal(loc=0, scale=1, size=1000)


def create_histogram():
    fig = px.histogram(x_rand, nbins=100)

    return fig


offline.plot(create_histogram())
