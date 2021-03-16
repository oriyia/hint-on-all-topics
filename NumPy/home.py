import numpy as np
essay = np.array([["pass", "not pass", "not pass", "not pass", "pass"],
        ["not pass", "not pass", "not pass", "not pass", "not pass"],
        ["pass", "not pass", "pass", "not pass", "not pass"]])

u = len([y for str in essay for y in str if y == 'pass'])
print(u)