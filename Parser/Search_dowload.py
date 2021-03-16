import requests as rq
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError
import re
import scrapy

# блок данных где содержится календарь событий
# 'название компании' + 'оффициальный сайт компании' - поисковый запрос в поисковике

URLS = ['https://yandex.ru/company/', 'https://www.nornickel.ru/', 'https://www.gazprom.ru/',
        'https://www.detmir.ru/', 'https://www.tatneft.ru/?lang=ru']
# URL = 'https://www.aeroflot.ru/ru-ru'
# URL = 'https://www.tatneft.ru/?lang=ru'
# URL = 'https://www.tatneft.ru/aktsioneram-i-investoram/raskritie-informatsii/ezhekvartalnie-otcheti?lang=ru'
URL = URLS[-1]
HEADERS = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/84.0.4147.89 Safari/537.36',
           'accept': '*/*'}
company = ['аэрофлот', 'татнефть']
value = company[1] + ' официальный сайт компании'
search_company = 'https://www.google.com/search?q=%D0%B0%D1%8D%D1%80%D0%BE%D1%84%D0%BB%D0%BE%D1%82+%D0%BE%D1%84%D0%B8%D1%86%D0%B8%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9+%D1%81%D0%B0%D0%B9%D1%82+%D0%BA%D0%BE%D0%BC%D0%BF%D0%B0%D0%BD%D0%B8%D0%B8&oq=%D0%B0%D1%8D%D1%80%D0%BE%D1%84%D0%BB%D0%BE%D1%82+%D0%BE%D1%84&aqs=chrome.0.69i59l2j69i57j0l4j69i60.8918j0j1&sourceid=chrome&ie=UTF-8'


def get_html(url, params=None):
    html = rq.get(url, headers=HEADERS, params=params)
    return html

def get_link_investor(html):
    link_site = BeautifulSoup(html.text, 'html.parser')
    for link in link_site.find_all('a'):
        if re.findall(r'\bинвесто\w+|\binvesto\w+', link.get_text(), flags=re.IGNORECASE):
            print(link.attrs['href'])
            return link.attrs['href']

def search_file(html):
    link_site = BeautifulSoup(html.text, 'html.parser')
    link_file = link_site.find_all('a', {'href': re.compile('\.pdf|\.zip')})[0].attrs['href']
    print(link_file)
    return link_file


def clear_url(url):
    for x in re.finditer(r'\.com/|\.ru/|\.org/|\.edu/|\.net/', url):
        return url[:x.end()]

def redactor_link(link_investor, clear_url):
    if link_investor[0] == '/':
        full_link = 'https://' + (clear_url + link_investor)[8:].replace('//', '/')
    else:
        full_link = link_investor
    print(full_link)
    return full_link

# for URL in URLS:
#     html = get_html(URL)
#     clear = clear_url(URL)
#     link = get_link_investor(html)
#     redactor_link(link, clear)
html = get_html(URL)
clear = clear_url(URL)
link = get_link_investor(html)
# link = search_file(html)
new_link = redactor_link(link, clear)

# html2 = get_html(new_link)
# clear2 = clear_url(new_link)
# link2 = search_file(html2)
# new_link2 = redactor_link(link2, clear2)

