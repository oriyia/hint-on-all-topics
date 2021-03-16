import scrapy


class PostsSpider(scrapy.Spider):
    name = 'first_project'
    start_urls = ['https://www.aeroflot.ru/ru-ru']

    def parse(self, response):
        page = response.xpath("//div[@class='head-r']/h2/text()").get()
        print(page)

