import numpy as np
import pandas as pd


arr = pd.DataFrame([[1, 2, 3], [1, 2, 3]])
print(arr)
arr.columns = ['hfp', 'fa', 'asdf']
print(arr)
arr.columns = ['asdf', 'asdfas', 'sdafdas']
print(arr)