import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

PATH = os.path.dirname(__file__)

data = np.load(os.path.join(PATH, 'main_rbfNonLinearYFullTrain10000scaledSig2.npz'))
y = data['train_y']
z = data['train_z']
print(y.shape, z.shape)

df = pd.DataFrame([y,z]).T
sns.pairplot(df)
plt.show()
