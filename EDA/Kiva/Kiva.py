import plotly.offline as offline
offline.init_notebook_mode()
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import missingno as msno
from datetime import datetime, timedelta
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import os
import matplotlib.pyplot as plt

# РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ EDA

pd.set_option('display.max_columns', 15)



df_kiva_loans = pd.read_csv('kiva_loans.csv')
df_mpi = pd.read_csv('kiva_mpi_region_locations.csv')
msno.bar(df_kiva_loans)  # пропущенные значения в датасете
msno.matrix(df_kiva_loans)  # структурное содеражание пропущенных значений

# print(df_kiva_loans.describe(include=[np.number]))  # сводка данных по цифрам
# print(df_kiva_loans.describe(include=[np.object]))  # данные по тексту (кол-во, сколько уникальных значений,
# самое популярное и сколько раз оно было, и так по каждому столбцу
country = df_kiva_loans['country'].value_counts()[df_kiva_loans['country'].value_counts(normalize=True)> 0.005]

# fig = go.Figure(data=[go.Histogram(y=country.index, x=country.values, histnorm='probability density')])
# fig = go.Figure()
# fig.add_trace(go.Bar(y=country.index, x=country.values, orientation='h'))

# print(df_mpi.head())
df_mpi_group = df_mpi.groupby(['ISO', 'country', 'world_region'])['MPI'].mean().fillna(0).reset_index()
df_kiva_loans = df_kiva_loans.merge(df_mpi_group, how='left', on='country')
regions = df_kiva_loans['world_region'].value_counts()
fig = go.Figure()
# fig.add_trace(go.Bar(y=regions.index, x=regions.values, orientation='h'))
# offline.plot(fig)

df_kiva_loans['borrower_genders'] = [obj if obj in ['female', 'male'] else 'group'
                                     for obj in df_kiva_loans['borrower_genders']]
borrowers = df_kiva_loans['borrower_genders'].value_counts()
pull = [0] * len(borrowers)
pull[borrowers.tolist().index(borrowers.max())] = 0.02
# fig.add_trace(go.Pie(values=borrowers, labels=borrowers.index, pull=pull, hole=0.8))

# count_gender = df_kiva_loans[['country', 'borrower_genders', ''].value_counts()
# df_kiva_loans['count_gender'] = df_kiva_loans.groupby('country')['borrower_genders'].value_counts()
# print(df_kiva_loans[['country', 'borrower_genders']])
country_gender = df_kiva_loans.groupby('country')['borrower_genders'].value_counts()
country_gender = country_gender.sort_values(by=country_gender.values)
# country_gender = country_gender[country_gender.values > country_gender.values.mean()].sort_values()
index1 = [x[0] for x in country_gender.index]
index2 = [x[1] for x in country_gender.index]
# print(country_gender['Indonesia'])
# print(index2)
# fig.add_trace(go.Bar(y=count_gender.index, x=count_gender))
# fig.update_layout(barmode='stack')
# offline.plot(fig)
# fig = px.bar(count_gender, x=count_gender.index, y=count_gender.values, barmode='group')

fig = px.bar(country_gender, y=index1, x=country_gender.values, color=index2, barmode='relative', orientation='h')
fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                  barmode='stack')
offline.plot(fig)
