a = {60: {99: 4, 36: 5}, 50: {99: 4, 36: 5}}
print(a)
b= sorted(a.items(), key=lambda item: item[0])
print(b)