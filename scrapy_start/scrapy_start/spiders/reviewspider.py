# -*- coding: utf-8 -*-
from scrapy import Field, Item
from scrapy.selector import Selector
from scrapy import Spider
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import time
# from product_scraper.items import Product

# class ReviewspiderSpider(scrapy.Spider):
#     name = 'reviewspider'
#     allowed_domains = ['www.tinkoff.ru']
#     start_urls = ['https://www.tinkoff.ru/summary/?auth=null']
#
#     def parse(self, response):
#         a = response.xpath("//h2[@class='DashboardTitle__title_3iJcF']/span/text()").get()
#         print(a)
df = pd.DataFrame()

driver = webdriver.Chrome()

driver.get('https://www.tinkoff.ru/invest/etfs?orderType=Desc&sortType=ByEarnings&start=0&end=38')
time.sleep(3)
page = driver.page_source
soup = BeautifulSoup(page, features='lxml')
items = soup.find_all('tr', class_='Table__row_e5625L Table__row_clickable_f5625L')

for item in items:
    ticker = item.find('div', 'Caption__subcaption_b1obXi').get_text(strip=True)
    price = item.find('div', 'SecurityColumn__cellPriceSecurities_g34puz').get_text(strip=True)
    for x in ('₽', '$', '€'):
        if x in price:
            currency = x
            price = price[:-1]
    if ',' in price:
        price = price.replace(',', '.')
    price = int(price)
    print(ticker, price, currency)


driver.close()
'''
class Article(Item):
    title = Field()

class ReviewspiderSpider(Spider):
    name = 'reviewspider'
    allowed_domains = ['www.tinkoff.ru']
    start_urls = ['https://www.tinkoff.ru/api/front/dashboard/v1/load-dashboard?wuid=b232ba2f94a94c0f93d386621295a9ed&dmpId=4ab0a9ab-2dbb-4ad5-86f9-8a4746f0308a&pcId=37656906&client=heavy&timezone=Europe%2FDublin&timeout=300&dashboard=ib']

    def parse(self, response):
        item = Article()
        title = response.xpath("//div[@class='Item__balance_p2f9X8']/span/text()").extract()
        print(title)
        item['title'] = title
        return item'''