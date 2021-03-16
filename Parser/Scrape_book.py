from urllib.request import urlopen
import requests as rq
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup
content = []

# find_all(tag, attributes, recursive=True(внутри глубже), text='текст который ищем', limit=1, keywords)
# find(tag, attributes, recursive=True(внутри глубже), text='текст который ищем', keywords).далее может быть
# .children()  извлечение только дочерник тегов
# .descendants()  извление всех потомков
# .next_siblings()  извление следующих одноуровневых элементов (но без того который мы указали)
# .next_sibling()  извлечение следующего одноуровневого элемента
# .previous_sibling()  предыдущий
# .parents/parent  поиск родителей (родителя)

# [A-Za-z0-9\._+]+@[A-Za-z]+\.(com|ru|org|edu|net)  # регулярное выражение
# * - ищет 0 или больше символов
# + - ищет 1 или больше символов
# [] - ищет любое совпадение с символом
# .find_all('img', {'scr': re.compile('\.\.\/img\/gifts/img.*\.jpg')}) -- ../img/gifts/img1.jpg и т.д.
# page.find_all(lambda tag: len(tag.attrs) == 2)



def getTitle(url):
    try:
        html = rq.get(url)
    except HTTPError as h:
        return None
    try:
        page = BeautifulSoup(html.text, 'html.parser')
        contents = page.find_all('div', class_='_93444fe79c--card--_yguQ')
        for x in contents:
            content.append(x.find('div', 'c6e8ba5398--header--1dF9r').get_text())
    except AttributeError as atr:
        return None
    return content

title = getTitle('https://irkutsk.cian.ru/kupit-kvartiru-1-komn-ili-2-komn/')
if title == None:
    print('title could not be found.')
else:
    print(content)

