import requests as rq
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import plotly.offline as offline
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as make_subplots
import csv

pd.set_option('display.max_columns', 7)
desired_width = 500
pd.set_option('display.width', desired_width)


URL = 'https://izhevsk.cian.ru/kupit-kvartiru-1-komn-ili-2-komn/'
HEADERS = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/84.0.4147.89 Safari/537.36',
           'accept': '*/*'}
FILE = 'House.csv'

def get_html(url, params=None):
    r = rq.get(url, headers=HEADERS, params=params)
    return r

def get_pages_count(html):
    soup = BeautifulSoup(html, 'html.parser')
    pagination = soup.find_all('li', class_='_93444fe79c--list-item--2KxXr')
    if pagination:
        return int(pagination[-2].get_text())
    else:
        return 1

def save_file(items, path):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Цена', 'Цена_за_кв.м', 'кв.м', 'Ссылка', 'Заголовок', 'Подпись'])
        for item in items:
            writer.writerow([item['Цена'], item['Цена_за_кв.м'], item['кв.м'], item['Ссылка:'],
                             item['Заголовок'], item['Подпись']])


house = []
def get_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('div', class_='c6e8ba5398--main--332Qx')

    for item in items:
        price = int(item.find('div', 'c6e8ba5398--header--1dF9r').get_text(strip=True)[:-2].replace(' ', ''))
        price_m = int(item.find('div', 'c6e8ba5398--term--3kvtJ').get_text(strip=True)[:-5].replace(' ', ''))
        house.append({'Цена': price,
                      'Цена_за_кв.м': price_m,
                      'кв.м': round(price / price_m, 2),
                      'Ссылка:': item.find('a', 'c6e8ba5398--header--1fV2A').get('href'),
                      'Заголовок': item.find('a', 'c6e8ba5398--header--1fV2A').get_text(),
                      'Подпись': item.find('div', 'c6e8ba5398--address-links--1tfGW').get_text(strip=True)
                      })

def parse():
    html = get_html(URL)
    if html.status_code == 200:
        pages_count = get_pages_count(html.text)
        for page in range(1, pages_count + 1):
            print(f'парсинг страницы {page} из {pages_count}')
            html = get_html(URL, params={'page': page})
            get_content(html.text)
        save_file(house, FILE)
    else:
        print('Error')

parse()
table = pd.DataFrame(house)
table.sort_values(['Цена_за_кв.м', 'кв.м'], ascending=False, inplace=True)
table.reset_index(inplace=True, drop=True)
# print(table[table['Цена за кв.м'] == table['Цена за кв.м'].min()])
print(table['Цена_за_кв.м'].head(200))

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=table['Цена за кв.м'], y=table['кв. м'], mode='lines+markers'))
fig = px.line(table, x='Цена_за_кв.м', y='кв.м')

# offline.plot(fig)