import numpy as np
import pandas as pd

x = np.random.rand(3, 3)
df = pd.DataFrame(x)
df.to_csv('output.csv', index=False)