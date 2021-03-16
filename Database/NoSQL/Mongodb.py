import pymongo as mg
import pandas as pd


conn = 'mongodb://students:X63673t47Gl03Sq@dsstudents.skillbox.ru:27017/?authSource=movies'

connect = mg.MongoClient(conn)

db = connect['movies']

collection = db.list_collection_names()  # список коллекций
print(collection)

tags = db['tags']  # сохраняем под tags одну из коллекций

print(tags.find_one())  # вывод первого документа коллекции (json)
print(tags.find().count())  # количество документов в коллекции
head = tags.find().limit(5)  # первые 5 документов коллекции (объект Cursor - итератор, а не список словарей)
# к head можно обратиться только один раз, нужно сохранить его в другую переменную
my_list = list(head)
df = pd.DataFrame(my_list)

# ФИЛЬТРЫ
head = tags.find({'id': {'$eq': 4290}}, {'_id': True})  # id - необходимый столбец, равный 4290,выводится только _id



