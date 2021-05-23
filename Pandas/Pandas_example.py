import numpy as np
import pandas as pd


# ДОБАВЬ ЖЕ ЧТО-НИБУДЬ ПОЛЕЗНОЕ
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['one', 'two', 'three'], index=['a', 'b', 'c'])
print(df)
print(df.iloc[1, 2])
print(df[['a', 'b']][['one', 'two']])
