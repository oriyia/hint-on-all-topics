import asyncio
import tinvest
from openapi_client import openapi
import pandas as pd
import numpy as np
import datetime
from pytz import timezone
import plotly.offline as offline
# offline.init_notebook_mode()
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

pd.set_option('display.max_columns', 14)
desired_width = 3000
pd.set_option('display.width', desired_width)

token = 't.H7LHOHLUX6EaC_KG25vb5WJJIVlJ1PZwKgz5_t-Ta0O23FX-XpT31JAXIsK_pcmzXZ7EFhgCNOVDWpvVJ8xeVQ'
broker_id = 22
ucc_id = 2011429186
client = openapi.api_client(token)

# Загрузка текущих актуальных данных
a = {1, 2, 3}


# загрузка текущего портфеля
def update_info():
    global pf_ucc
    pf_ucc = client.portfolio.portfolio_get(broker_account_id=ucc_id)

    # загрузка курса валют
    def exchange_rates():
        global usd, eur
        try:
            usd = client.market.market_orderbook_get('BBG0013HGFT4', 1).payload.asks[0].price
        except:
            usd = client.market.market_orderbook_get('BBG0013HGFT4', 1).payload.close_price
        try:
            eur = client.market.market_orderbook_get('BBG0013HJJ31', 1).payload.asks[0].price
        except:
            eur = client.market.market_orderbook_get('BBG0013HJJ31', 1).payload.close_price

    # загрузка кэша
    def free_money():
        global cash_rub, cash_usd, cash_eur, cash_all
        cash_rub = client.portfolio.portfolio_currencies_get(broker_account_id=ucc_id).payload.currencies[0].balance
        cash_usd = client.portfolio.portfolio_currencies_get(broker_account_id=ucc_id).payload.currencies[1].balance
        cash_eur = client.portfolio.portfolio_currencies_get(broker_account_id=ucc_id).payload.currencies[2].balance
        cash_all = cash_rub + cash_usd * usd + cash_eur * eur

    exchange_rates()
    free_money()

    # создание текущего портфеля
    global df
    df = pd.DataFrame(columns=['ticker', 'value', 'value_all', 'currency', 'currency_real', 'region', 'branch',
                               'developed', 'composition', 'balance', 'isin', 'change', 'instrument_type'])
    for i, position in enumerate(pf_ucc.payload.positions):
        try:
            if position.instrument_type == 'Currency':
                continue

            currency = position.average_position_price.currency
            balance = position.balance
            figi = position.figi
            ticker = position.ticker
            isin = position.isin
            change = position.expected_yield.value
            instrument_type = position.instrument_type
            try:
                value = round(client.market.market_orderbook_get(figi, 1).payload.asks[0].price, 4)
            except:
                value = round(client.market.market_orderbook_get(figi, 1).payload.close_price, 4)
            if currency == 'USD':
                value *= usd
            value_all = round(value * balance, 4)
            currency_real = etfs_all.get(ticker, [0, 1, currency])[2]
            region = etfs_all.get(ticker, ['Россия', None, None])[0]
            developed = etfs_all.get(ticker, [0, 'no', None])[1]
            composition = etfs_all.get(ticker, [None, None, None, instrument_type])[3]
            branch = etfs_all.get(ticker, [None, None, None, None, None])[4]

            df.loc[i] = [ticker,
                         value,
                         value_all,
                         currency,
                         currency_real,
                         region,
                         branch,
                         developed,
                         composition,
                         balance,
                         isin,
                         change,
                         instrument_type]
        except Exception as ex:
            print(ex)
            break
    print(df)


etfs_all = {'TECH': ['Америка', 'yes', 'USD', 'Stock', 'IT', 'TECHA.ME'],
            'TMOS': ['Россия', 'no', 'RUB', 'Stock', None, 'TMOSA.ME'],
            'FXUS': ['Америка', 'yes', 'USD', 'Stock', None, 'FXUS.ME'],
            'FXCN': ['Китай', 'no', 'USD', 'Stock', None, 'FXCN.ME'],
            'SBMX': ['Россия', 'no', 'RUB', 'Stock', None, 'SBMX.ME'],
            'FXGD': ['Мир', 'yes', 'USD', 'Gold', None, 'FXGD.ME'],
            'TRUR': ['Россия', 'no', 'RUB', 'All', None, 'TRURA.ME'],
            'VTBE': ['Развив-ся', 'no', 'USD', 'Stock', None, 'VTBEM.ME'],
            'FXRL': ['Россия', 'no', 'RUB', 'Stock', None, 'FXRL.ME'],
            'FXIT': ['Америка', 'yes', 'USD', 'Stock', 'IT', 'FXIT.ME'],
            'TGLD': ['Мир', 'yes', 'USD', 'Gold', None, 'TGLDA.ME'],
            'TUSD': ['Америка', 'yes', 'USD', 'All', None, 'TUSDA.ME'],
            'TEUR': ['Европа', 'yes', 'EUR', 'All', None, 'TEURA.ME'],
            'SBRB': ['Россия', 'no', 'RUB', 'Bond', 'Corp', 'SBRBA.ME'],
            'SBGB': ['Россия', 'no', 'RUB', 'Bond', 'OFZ', 'SBGBA.ME'],
            'SBCB': ['Россия', 'no', 'USD', 'Bond', 'Corp', 'SBCBA.ME'],
            'SBSP': ['Америка', 'yes', 'USD', 'Stock', None, 'SBSPA.ME'],
            'VTBG': ['Мир', 'yes', 'USD', 'Gold', None, 'VTBGO.ME'],
            'VTBM': ['Россия', 'no', 'RUB', 'Bond', 'OFZ', 'VTBMM.ME'],
            'VTBB': ['Россия', 'no', 'RUB', 'Bond', 'Copr', 'VTBBA.ME'],
            'VTBH': ['Америка', 'yes', 'USD', 'Bond', 'Corp', 'VTBHY.ME'],
            'VTBU': ['Россия', 'no', 'USD', 'Bond', 'Corp', 'VTBUE.ME'],
            'VTBX': ['Россия', 'no', 'RUB', 'Stock', None, 'VTBXE.ME'],
            'VTBA': ['Америка', 'yes', 'USD', 'Stock', None, 'VTBSN2.ME'],
            'AKMB': ['Россия', 'no', 'RUB', 'Bond', 'All', 'AKMBA.ME'],
            'AKNX': ['Америка', 'yes', 'USD', 'Stock', 'IT', 'AKNXA.ME'],
            'AKEU': ['Европа', 'yes', 'EUR', 'Stock', None, 'AKEUA.ME'],
            'AKSP': ['Америка', 'yes', 'USD', 'Stock', None, 'AKSPA.ME'],
            'RUSB': ['Россия', 'no', 'USD', 'Bond', 'All', 'RUSB.ME'],
            'RUSE': ['Россия', 'no', 'USD', 'Stock', None, 'RUSE.ME'],
            'FXTB': ['Америка', 'yes', 'USD', 'Bond', 'OFZ', 'FXTB.ME'],
            'FXMM': ['Америка', 'yes', 'RUB', 'Bond', 'OFZ', 'FXMM.ME'],
            'FXRU': ['Россия', 'no', 'USD', 'Bond', 'Corp', 'FXRU.ME'],
            'FXRB': ['Россия', 'no', 'RUB', 'Bond', 'Corp', 'FXRB.ME'],
            'FXDE': ['Германия', 'yes', 'EUR', 'Stock', None, 'FXDE.ME'],
            'FXWO': ['Мир', 'yes/no', 'USD', 'Stock', None, 'FXWO.ME'],
            'FXRW': ['Мир', 'yes/no', 'RUB', 'Stock', None, 'FXRW.ME']}

update_info()
# ВЫВОД СКОЛЬКО ДЕНЕГ
# НАЛИЧНЫЕ

# rules
rub_ref = 10  # доля рублей
usd_ref = 80  # доля доллара
eur_ref = 10

stock_ref = 70  # доля акций
bond_ref = 20  # доля облигаций
gold_ref = 10  # доля золота

stock_dev_ref = 50  # доля акций развитых стран
stock_undev_ref = 50  # доля акций развивающихся стран

stock_usa_ref = 70  # доля акций америки
stock_eur_ref = 30  # доля акций европы

stock_rus_ref = 20  # доля акций россии
stock_ch_ref = 40
stock_all_ref = 40

bond_dev_ref = 80  # доля облигаций развитых стран
bond_undev_ref = 20  # доля облигаций развивающихся стран

bond_usa_ref = 50
bond_eur_ref = 50

bond_rus_ref = 30
bond_ch_ref = 40
bond_all_ref = 30


def statistic():
    money_invest = df.value_all.sum()
    count_invest = df.shape[0]
    money_all = cash_all + money_invest
    print(f'Наличные всего: {round(cash_all, 2)} руб.')
    print(f'Рубли: {cash_rub} руб. ({round(cash_rub / cash_all * 100, 1)}%), '
          f'Доллары: {cash_usd} дол. ({round(cash_usd * usd, 2)} руб. {round(cash_usd * usd / cash_all * 100, 1)}%), '
          f'Евро: {cash_eur}({round(cash_eur * eur, 2)} руб. {round(cash_eur * eur / cash_all * 100, 1)}%)')
    print(f'Инвестированно: {round(money_invest, 2)} руб., (всего пунктов - {count_invest})')
    print(f'Всего на счете: {round(money_all, 2)} руб.')

    print('\n    ЗАИНВЕСТИРОВАННО')
    rub_inv = df[df.currency_real == 'RUB']['value_all'].sum()
    composition_rub = list(df[df.currency_real == 'RUB'].ticker)
    usd_inv = df[df.currency_real == 'USD']['value_all'].sum()
    composition_usd = list(df[df.currency_real == 'USD'].ticker)
    eur_inv = df[df.currency_real == 'EUR']['value_all'].sum()
    composition_eur = list(df[df.currency_real == 'EUR'].ticker)
    rub_pr = int(rub_inv / money_invest * 100)
    usd_pr = int(usd_inv / money_invest * 100)
    eur_pr = int(eur_inv / money_invest * 100)
    rub_deviation = rub_pr - rub_ref
    usd_deviation = usd_pr - usd_ref
    eur_deviation = eur_pr - eur_ref
    print(f'Рублей: {round(rub_inv, 2)}  {rub_pr}% ({rub_deviation}%)')
    print(f'    {composition_rub}')
    print(f'Долларов: {round(usd_inv, 2)}  {usd_pr}% ({usd_deviation}%)')
    print(f'    {composition_usd}')
    print(f'Евро: {round(eur_inv, 2)}  {eur_pr}% ({eur_deviation}%)')
    print(f'    {composition_eur}')

    print('\n    СООТНОШЕНИЕ ТИПОВ БУМАГ')
    stock = round(df[df.composition == 'Stock']['value_all'].sum(), 2)
    composition_stock = list(df[df.composition == 'Stock'].ticker)
    bond = round(df[df.composition == 'Bond']['value_all'].sum(), 2)
    composition_bond = list(df[df.composition == 'Bond'].ticker)
    gold = round(df[df.composition == 'Gold']['value_all'].sum(), 2)
    composition_gold = list(df[df.composition == 'Gold'].ticker)
    stock_pr = int(stock / money_invest * 100)
    bond_pr = int(bond / money_invest * 100)
    gold_pr = int(gold / money_invest * 100)
    print(f'Акции: {stock}  {stock_pr}% ({stock_pr - stock_ref}%)   {composition_stock}')
    print(f'Облигации: {bond}  {bond_pr}% ({bond_pr - bond_ref}%)   {composition_bond}')
    print(f'Золото: {gold}  {gold_pr}% ({gold_pr - gold_ref}%)    {composition_gold}')

    print('\n    АКЦИИ Соотношение развитых и развивающихся стран')
    stock_dev = round(df[(df.developed == 'yes') & (df.composition == 'Stock')]['value_all'].sum(), 2)
    stock_undev = round(df[(df.developed == 'no') & (df.composition == 'Stock')]['value_all'].sum(), 2)
    stock_dev_pr = int(stock_dev / stock * 100)
    stock_undev_pr = int(stock_undev / stock * 100)
    print(f'Развитые: {stock_dev}  {stock_dev_pr}% ({stock_dev_pr - stock_dev_ref}%)')
    print(f'Неразвитые: {stock_undev}  {stock_undev_pr}% ({stock_undev_pr - stock_undev_ref}%)')

    def dev():
        print('\n____ АКЦИИ Доли в развитых рынках ____')
        stock_usa = round(df[(df['region'] == 'Америка') & (df['developed'] == 'yes') & (df['composition'] == 'Stock')][
                              'value_all'].sum(), 2)
        stock_eua = round(df[(df['region'] == 'Европа') & (df['developed'] == 'yes') & (df['composition'] == 'Stock')][
                              'value_all'].sum(), 2)

        stock_usa_pr = round(stock_usa / stock_dev * 100, 2)
        stock_eua_pr = round(stock_eua / stock_dev * 100, 2)
        print(f'Америка: {stock_usa}  {stock_usa_pr}% ({stock_usa_pr - stock_usa_ref}%)')
        print(f'Европа: {stock_eua}  {stock_eua_pr}% ({stock_eua_pr - stock_eur_ref}%)')

    dev()

    def undev():
        print('\n____ АКЦИИ Доли в неразвитых рынках ____')
        stock_rus = round(df[(df['region'] == 'Россия') & (df['developed'] == 'no') & (df['composition'] == 'Stock')][
                              'value_all'].sum(), 2)
        stock_ch = round(df[(df['region'] == 'Китай') & (df['developed'] == 'no') & (df['composition'] == 'Stock')][
                             'value_all'].sum(), 2)
        stock_all = round(
            df[(df['region'] == 'Развив-ся') & (df['developed'] == 'no') & (df['composition'] == 'Stock')][
                'value_all'].sum(), 2)

        stock_rus_pr = round(stock_rus / stock_undev * 100, 2)
        stock_ch_pr = round(stock_ch / stock_undev * 100, 2)
        stock_all_pr = round(stock_all / stock_undev * 100, 2)
        print(f'Россия: {stock_rus}  {stock_rus_pr}% ({stock_rus_pr - stock_rus_ref}%)')
        print(f'Китай: {stock_ch}  {stock_ch_pr}% ({stock_ch_pr - stock_ch_ref}%)')
        print(f'Мир: {stock_all}  {stock_all_pr}% ({stock_all_pr - stock_all_ref}%)')

    undev()

    def bond_dev_undev():
        print('\n____ ОБЛИГАЦИИ Соотношение развитых и развивающихся стран ____')
        global bond_dev
        global bond_undev
        bond_dev = round(df[(df['developed'] == 'yes') & (df['composition'] == 'Bond')]['value_all'].sum(), 2)
        bond_undev = round(df[(df['developed'] == 'no') & (df['composition'] == 'Bond')]['value_all'].sum(), 2)
        bond_dev_pr = int(bond_dev / bond * 100)
        bond_undev_pr = int(bond_undev / bond * 100)
        print(f'Развитые: {bond_dev}  {bond_dev_pr}% ({bond_dev_pr - bond_dev_ref}%)')
        print(f'Неразвитые: {bond_undev}  {bond_undev_pr}% ({bond_undev_pr - bond_undev_ref}%)')

    if bond != 0:
        bond_dev_undev()

    def bond_dev_otn():
        print('\n____ ОБЛИГАЦИИ Доли в развитых рынках ____')
        bond_usa = round(df[(df['region'] == 'Америка') & (df['developed'] == 'yes') & (df['composition'] == 'Bond')][
                             'value_all'].sum(), 2)
        bond_eua = round(df[(df['region'] == 'Европа') & (df['developed'] == 'yes') & (df['composition'] == 'Bond')][
                             'value_all'].sum(), 2)

        bond_usa_pr = round(bond_usa / bond_dev * 100, 2)
        bond_eua_pr = round(bond_eua / bond_dev * 100, 2)
        print(f'Америка: {bond_usa}  {bond_usa_pr}% ({bond_usa_pr - bond_usa_ref}%)')
        print(f'Европа: {bond_eua}  {bond_eua_pr}% ({bond_eua_pr - bond_eur_ref}%)')

    if bond != 0:
        bond_dev_otn()

    def bond_undev_otn():
        print('\n____ ОБЛИГАЦИИ Доли в неразвитых рынках ____')
        bond_rus = round(df[(df['region'] == 'Россия') & (df['developed'] == 'no') & (df['composition'] == 'Stock')][
                             'value_all'].sum(), 2)
        bond_ch = round(df[(df['region'] == 'Китай') & (df['developed'] == 'no') & (df['composition'] == 'Stock')][
                            'value_all'].sum(), 2)
        bond_all = round(df[(df['region'] == 'Развив-ся') & (df['developed'] == 'no') & (df['composition'] == 'Stock')][
                             'value_all'].sum(), 2)

        bond_rus_pr = round(bond_rus / bond_undev * 100, 2)
        bond_ch_pr = round(bond_ch / bond_undev * 100, 2)
        bond_all_pr = round(bond_all / bond_undev * 100, 2)
        print(f'Россия: {bond_rus}  {bond_rus_pr}% ({bond_rus_pr - bond_rus_ref}%)')
        print(f'Китай: {bond_ch}  {bond_ch_pr}% ({bond_ch_pr - bond_ch_ref}%)')
        print(f'Мир: {bond_all}  {bond_all_pr}% ({bond_all_pr - bond_all_ref}%)')

    if bond != 0:
        bond_undev_otn()


statistic()
#
#
# # for op in ops.payload.operations: # Перебираем операции
# #     print(op.figi) # figi всегда берем из операции
# #     print(op.operation_type)   # и тип операции тоже
# #     if op.trades == None:      # Если биржевых сделок нет
# #         print('price:', op.price)       # Берем из операции цену бумаги
# #         print('pay.ment:', op.payment)   # Сумму платежа
# #         print('quantity:', op.quantity) # И количество бумаг
# #     else:
# #         for t in op.trades:                   # А если есть сделки - то перебираем их
# #             print('price:', t.price)          # И берем данные из них
# #             print('quantity:', t.quantity)
# #     print('--------------')
#
# # date = datetime.today()
# # перевод в строку
# # date.strftime('%d-%m-%Y %H:%M') %d - день, %m - мес, %Y - год, %H - час (24), %M - мин, %S - сек,
# # date.strftime('%x')  %с - время и дата, %x - дата, %X - время
# # date.strftime('%A') - %A - полное название дня недели, %a - сокр, %s - в виде числа, МЕСЯЦЫ: %B, %b, %m
# # date_string = '21 September. 1999'
# # dateobject = datetime.strptime(date_string, '%d %B. %Y')
# # duration_time = datetime.timedelta(days=12, seconds = 33)
#
#
# def f_sdelki(d1, d2):
#     # загрузка сделок
#     return client.operations.operations_get(_from=d1.isoformat(), to=d2.isoformat(), broker_account_id=2011429186)
#
#
# def time_zona():
#     # устанавливаем дату и время с которых будем скачивать данные
#     open_file()  # загружаем файлы
#     # '2020-02-29 00:01:00' стандарт записи
#     try:
#         last_data = price_portfel.date.iloc[-1]  # получаем последнюю дату загрузки
#         t = datetime.datetime.strptime(last_data, '%Y-%m-%d %H:%M:%S')  # преобразовываем дату-строку в дату-объект
#         last_data = datetime.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second + 1,
#                                       tzinfo=timezone('Europe/Moscow'))
#     except IndexError:
#         last_data = datetime.datetime(2019, 10, 14, 0, 0, 0, tzinfo=timezone('Europe/Moscow'))
#     # выводим временные рамки
#     return last_data, datetime.datetime.now(tz=timezone('Europe/Moscow'))
#
#
# def open_file():
#     global portfel, price_portfel, cash
#     try:
#         portfel = pd.read_csv('portfel.csv')
#         price_portfel = pd.read_csv('price_portfel.csv')
#         cash = pd.read_csv('cash.csv')
#     except FileNotFoundError:
#         portfel = pd.DataFrame(columns=['figi', 'count', 'price'])
#         price_portfel = pd.DataFrame([['2019-10-14 00:00:00', 0]], columns=['date', 'price'])
#         cash = pd.DataFrame([['2019-10-14 00:00:00', 0]], columns=['date', 'price'])
#     except PermissionError:
#         print(f'Ошибка! Файл открыт!')
#
#
# def save_file():
#     try:
#         portfel.to_csv('portfel.csv', index=False)
#         price_portfel.to_csv('price_portfel.csv', index=False)
#         cash.to_csv('cash.csv', index=False)
#     except PermissionError:
#         print(f'Ошибка! Файл используется другой программой!')
#
#
# def dobavlenie(operation, date=None, price=None, figi=None, count=None):
#     date_sr = date.strftime('%Y-%m-%d')  # дата для сравнения дат
#     date_save = date.strftime('%Y-%m-%d %H:%M:%S')  # дата для записи в файл
#     if operation == 'Buy':  # покупка
#         if figi in portfel.figi:  # если бумага с таким figi уже есть
#             portfel.loc[portfel['figi'] == figi, 'count'] += count
#         else:
#             portfel.loc[portfel.shape[0]] = [figi, count, price]
#         pereraschet(date)
#     elif operation == 'Sell':  # продажа
#         if figi in portfel.figi:  # если бумага с таким figi уже есть
#             portfel.loc[portfel['figi'] == figi, 'count'] -= count
#             if portfel.loc[portfel['figi'] == figi, 'count'] == 0:  # если бумага продана полностью
#                 portfel.drop(index=portfel[portfel['figi'] == figi].index[0], inplace=True)
#         pereraschet(date)
#     elif operation in ['PayIn', 'Dividend', 'Coupon']:  # пополнение
#         if date_sr == datetime.datetime.strptime(cash.date.iloc[-1], '%Y-%m-%d %H:%M:%S').strftime(
#                 '%Y-%m-%d'):  # если даты совпали
#             cash['price'][cash.shape[0] - 1] += price
#         else:
#             cash.loc[cash.shape[0]] = [date_save, cash['price'][cash.shape[0] - 1] + price]
#     elif operation == 3:  # вывод
#         if date_sr == datetime.datetime.strptime(cash.date.iloc[-1], '%Y-%m-%d %H:%M:%S').strftime(
#                 '%Y-%m-%d'):  # если  даты совпали
#             cash['price'][cash.shape[0] - 1] -= price
#         else:
#             cash.loc[cash.shape[0]] = [date_save, cash['price'][cash.shape[0] - 1] - price]
#     if operation in ['Buy', 'Sell']:
#         if date_sr == datetime.datetime.strptime(price_portfel.date.iloc[-1], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'):
#             price_portfel['price'][price_portfel.shape[0] - 1] = portfel['price'].sum()
#         else:
#             price_portfel.loc[price_portfel.shape[0]] = [date_save, portfel['price'].sum()]  # !!!!! добавь кэша по дате
#
#
# def pereraschet(date):
#     d1 = datetime.datetime(date.year, date.month, date.day, 11, 0, 0, tzinfo=timezone('Europe/Moscow'))
#     d2 = datetime.datetime(date.year, date.month, date.day, 16, 0, 0, tzinfo=timezone('Europe/Moscow'))
#     date_sr = date.strftime('%Y-%m-%d')  # дата для сравнения дат
#     date_now = datetime.datetime.now(tz=timezone('Europe/Moscow')).strftime('%Y-%m-%d')
#     print(d1, d2)
#     for fig in portfel.figi:
#         tic = client.market.market_search_by_figi_get(fig).payload.ticker
#         if client.market.market_search_by_figi_get(fig).payload.type == 'Etf':
#             if date_sr == date_now:
#                 ticker = client.market.market_search_by_figi_get(fig)
#                 value = round(pdr.get_quote_yahoo(etfs_all[tic][5]).price, 4)
#             else:
#                 try:
#                     value = round(pdr.get_data_yahoo_actions(etfs_all[tic][5], date).loc[date].Close, 4)
#                     portfel.loc[portfel['figi'] == fig, 'price'] = value * portfel.loc[portfel['figi'] == fig, 'count']
#                 except Exception as x:
#                     print(f'что=то не получилось {x}')
#         else:
#             tic = tic + '.ME'
#             if date_sr == date_now:
#                 ticker = client.market.market_search_by_figi_get(fig)
#                 value = round(pdr.get_quote_yahoo(tic).price, 4)
#             else:
#                 try:
#                     value = round(pdr.get_data_yahoo_actions(tic, date).loc[date].Close, 4)
#                     portfel.loc[portfel['figi'] == fig, 'price'] = value * portfel.loc[portfel['figi'] == fig, 'count']
#                 except Exception as x:
#                     print(f'что=то не получилось {x}')
#
#
# def up_data():
#     d1, d2 = time_zona()  # временные рамки
#     sdelki = f_sdelki(d1, d2)  # все сделки по установленному временному отрезку
#     print(f'asdfdfasdfasdfdasfdas {len(sdelki.payload.operations)}')
#     for x in reversed(sdelki.payload.operations):  # цикл по каждой сделке 196 всего 7 с чем-то[-25:]
#         print(x)
#         operation = x.operation_type  # тип операции
#         # ['PayIn', 'Buy', 'BrokerCommission', 'TaxDividend', 'Dividend', 'Sell', 'Coupon', 'PartRepayment', 'ServiceCommission']
#         if operation in ['Buy', 'Sell']:
#             figi = x.figi
#             price = x.trades[0].price
#             count = x.trades[0].quantity
#             date = x.trades[0].date
#             dobavlenie(operation, date, price, figi, count)
#         elif operation in ['PayIn', 'вывод']:
#             price = x.payment
#             date = x.date
#             dobavlenie(operation, date, price)
#         elif operation in ['Dividend', 'Coupon']:
#             price = x.payment
#             date = x.date
#             dobavlenie(operation, date, price)
#     print(portfel)
#     print(price_portfel)
#     print(cash)
#
#     save_file()
#
#
# up_data()
#
#
# def grafic(price_portfel, cash):
#     x = np.arange(0, 150, 1)
#
#     def gr(x):  # график эталонного пополнения в 50 000 в мес.
#         return 1666 * x
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=price_portfel.date, y=price_portfel.price, name='Все инвестиции'))
#     fig.add_trace(go.Scatter(x=cash.date, y=cash.price, name='Пополнения'))
#     fig.add_trace(go.Scatter(x=x, y=gr(x), name='Все инвестиции'))
#     offline.plot(fig)
