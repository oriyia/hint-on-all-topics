import plotly.offline as offline
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import numpy
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from collections import Counter



df_kiva_loans = pd.read_csv('kiva_loans.csv')
df_mpi = pd.read_csv('kiva_mpi_region_locations.csv')

df_kiva_loans['borrower_genders_n'] = [obj if obj in ['male', 'female'] else 'group'
                                     for obj in df_kiva_loans['borrower_genders']]

country_genders = df_kiva_loans.groupby(['country', 'borrower_genders_n'])['borrower_genders_n'].count().unstack().fillna(0)

country_genders['all'] = df_kiva_loans.groupby('country')['borrower_genders_n'].count()

country_genders['female_x'] = round(country_genders['female'] / country_genders['all'] * 100, 1)
country_genders['male_x'] = round(country_genders['male'] / country_genders['all'] * 100, 1)
country_genders['group_x'] = round(country_genders['group'] / country_genders['all'] * 100, 1)

country_genders_male = country_genders[country_genders['male_x'] >= 50].sort_values(by=['male_x'], ascending=False)
country_genders_group = country_genders[country_genders['group_x'] >= 50].sort_values(by=['group_x'], ascending=False)
# print(country_genders.head())

fig2 = px.bar(country_genders_group, x=country_genders_group.index, y=['group_x', 'male_x', 'female_x'],
              barmode='relative', title='Страны, с наибольшим количеством заемщиков - групп.')
fig3 = px.bar(country_genders_male, x=country_genders_male.index, y=['male_x', 'group_x', 'female_x'],
              barmode='relative', title='Страны, с наибольшим количеством заемщиков - групп.')

# offline.plot(fig3)

# ----------------------------------------------------------------------------------------------------
pd.set_option('max_colwidth', 800, 'display.max_columns', 10, 'display.width', 1000)
df_mpi_world_region = df_mpi.groupby(['country', 'world_region'])['MPI'].mean()
print(df_mpi_world_region)
df_kiva_loans = df_kiva_loans.merge(df_mpi_world_region, how='left', left_on='country', right_on=df_mpi_world_region.index)
print(df_kiva_loans.head())
funded_amount = df_kiva_loans[df_kiva_loans['funded_amount'] > 20000]
print(funded_amount[['country', 'use']])

print(funded_amount['country'].unique())
print(funded_amount['borrower_genders_n'].value_counts())
a = funded_amount[funded_amount['borrower_genders_n'] == 'group'][['country', 'borrower_genders']].dropna()
z = []
for x in a['borrower_genders']:
    z.extend(x.split(', '))
print(Counter(z))

print(funded_amount['country'].value_counts())
print(funded_amount.groupby(['country', 'borrower_genders_n']))

# fig = px.bar(funded_amount, x=funded_amount.country, y=funded_amount.funded_amount, color=funded_amount.borrower_genders_n)
# offline.plot(fig)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=funded_amount.country,
    y=funded_amount.funded_amount,
    # color=funded_amount.borrower_genders_n,
    mode="markers",
    marker=go.scatter.Marker(
        size=funded_amount.funded_amount/2000,
        opacity=0.6,
        colorscale="Viridis"
    )
))
# funded_amount = funded_amount[['country', 'borrower_genders_n', 'world_region']].dropna()
# fig = px.scatter(funded_amount, x=funded_amount.country, y=funded_amount.funded_amount,
#                  color=funded_amount.borrower_genders_n, size=funded_amount.funded_amount/1000)
# offline.plot(fig)


# a = df_mpi.groupby('world_region')['country'].unique()
# b = df_kiva_loans.groupby(['country', 'borrower_genders2'])['borrower_genders2'].count().unstack().fillna(0)
# # b['af'] = [1, 2 ,3 ,4, 5, 6]
# # print(b.index)
# c=[]
# for u in b.index:
#     for x in a.index:
#         if u in a.loc[x]:
#             c.append(x)
#             break
#     else:
#         c.append('Other region')
# b['world_region'] = c
# # print(b)
# n = b.groupby('world_region')[['female', 'group', 'male']].sum()
# # df_kiva_loans['world_region'] = []
# # print(n)
# fig3 = go.Figure()
# specs = [[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}],
#          [{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]]
# fig3 = make_subplots(rows=2, cols=4, specs=specs, subplot_titles=[n.index[x] for x in range(n.shape[0])])
# i=1
# j=1
# for m in range(n.shape[0]):
#     pull = [0.1 if x == max(n.iloc[m]) else 0 for x in n.iloc[m]]
#     fig3.add_trace(go.Pie(values=n.iloc[m], labels=n.columns, pull=pull, hole=0.6), i, j)
#     j += 1
#     if i==2 and j==4:
#         break
#     if j == 5:
#         i += 1
#         j = 1
#
# # fig3.update_layout(annotations=[dict(text=n.index[m], x=0.5, y=0.5, font_size=20, showarrow=False)])
# fig3.update_traces(textposition='inside', textinfo='percent+label')
# fig3.update_layout(margin=dict(l=0, t=20, b=0, r=0),
#                    legend_orientation='h',
#                    legend=dict(xanchor='left'))
# # offline.plot(fig3)
#
#
#
#
#
# gender_sector = df_kiva_loans.groupby(['sector', 'borrower_genders2'])['borrower_genders'].count().unstack().fillna(0)
# gender_sector['total'] = df_kiva_loans.groupby('sector')['borrower_genders'].count()
# # print(gender_sector.head(10))
#
#
#
#
#--------------------------------------------------------------------
# lender_countf = df_kiva_loans.groupby('lender_count')[['funded_amount',
#                                                        'term_in_months']].aggregate('mean')
# lender_countf.round({'funded_amount': 1})
# # print(lender_countf)
#
# fig4 = make_subplots(specs=[[{'secondary_y': True}]])
# fig4.add_trace(go.Scatter(x=lender_countf.index, y=lender_countf.funded_amount, name='yaxis1 data', mode='markers'), secondary_y=False)
# fig4.add_trace(go.Scatter(x=lender_countf.index, y=lender_countf.term_in_months, name='yaxis2 data', mode='markers'), secondary_y=True)
# fig4.update_yaxes(title_text='<b>primary</b> yaxis title', secondary_y=False)
# fig4.update_yaxes(title_text='<b>secondary</b> yaxis title', secondary_y=True)
# # offline.plot(fig4)
##--------------------------------------------------------------------
#
#
#
# mpi_st = df_mpi.groupby('country')['MPI'].mean()
# country_mpi = df_kiva_loans.groupby('country')[['funded_amount', 'term_in_months']].mean()
# country_mpi_s = country_mpi.merge(mpi_st, how='right', on='country').dropna()
# print(country_mpi_s)
#
#
# fig5 = make_subplots(specs=[[{'secondary_y': True}]])
# fig5.add_trace(go.Scatter(x=country_mpi_s.MPI, y=country_mpi_s.funded_amount, name='yaxis1 data', mode='markers'), secondary_y=False)
# fig5.add_trace(go.Scatter(x=country_mpi_s.MPI, y=country_mpi_s.term_in_months, name='yaxis2 data', mode='markers'), secondary_y=True)
# fig5.update_yaxes(title_text='<b>primary</b> yaxis title', secondary_y=False)
# fig5.update_yaxes(title_text='<b>secondary</b> yaxis title', secondary_y=True)
# offline.plot(fig5)