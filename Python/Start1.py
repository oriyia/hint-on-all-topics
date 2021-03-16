import csv

funnel_gender = {}
funnel_device = {}
funnel_by_month = {}
funnel_templation = {'home_page': 0, 'search_page': 0, 'payment_page': 0, 'payment_confirmation_page': 0}

with open('click_stream3.csv', mode='r') as csv_file:  # открываем файл
    csv_reader = csv.DictReader(csv_file)  # читаем файл

    for row in csv_reader:
        page = list(row.items())[1][1][2:]
        event_date = list(row.items())[2][1][:-3]
        device = list(row.items())[3][1]
        gender = list(row.items())[4][1]

        if gender not in funnel_gender:
            funnel_gender[gender] = funnel_device.copy()
        if device not in funnel_gender[gender]:
            funnel_gender[gender][device] = funnel_by_month.copy()
        if event_date not in funnel_gender[gender][device]:
            funnel_gender[gender][device][event_date] = funnel_templation.copy()

        if page == 'home_page':
            funnel_gender[gender][device][event_date]['home_page'] += 1
        elif page == 'search_page':
            funnel_gender[gender][device][event_date]['search_page'] += 1
        elif page == 'payment_page':
            funnel_gender[gender][device][event_date]['payment_page'] += 1
        else:
            funnel_gender[gender][device][event_date]['payment_confirmation_page'] += 1

for key1 in funnel_gender.keys():
    for key2 in funnel_gender[key1].keys():
        keys_event_data = sorted(list(funnel_gender[key1][key2].keys()))

        for key3 in keys_event_data:
            # операция присваивания для упрощения кода
            a1 = funnel_gender[key1][key2][key3]['home_page']
            b1 = funnel_gender[key1][key2][key3]['search_page']
            c1 = funnel_gender[key1][key2][key3]['payment_page']
            d1 = funnel_gender[key1][key2][key3]['payment_confirmation_page']
            # процент посетителей на разных страницах по отношению к home_page = '100%'
            search_page = round(b1/a1 * 100, 2)
            payment_page = round(c1/a1 * 100, 2)
            payment_confirmation_page = round(d1/a1 * 100, 2)
            # процент по отношению к предыдущей старнице
            payment_page_up = round(c1 / b1 * 100, 2)
            payment_confirmation_page_up = round(d1 / c1 * 100, 2)

            print(key1, key2, key3, a1, '  ', b1, f'({search_page}),   ', c1,
                  f'({payment_page})({payment_page_up}),   ',
                  d1, f'({payment_confirmation_page})({payment_confirmation_page_up}).')
        print('')
    print('')