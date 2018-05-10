import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../data/train.csv')
YOB=df.loc[:,'YOB'].as_matrix()
YOB=YOB[~np.isnan(YOB)]
fig = plt.figure('Histogram of YOB')
fig.suptitle('Histogram of YOB', fontsize=14, fontweight='bold')

num_bins = 15
plt.hist(YOB, num_bins, normed=1)
plt.xlim(1920,2010)
plt.xlabel('YOB')
plt.ylabel('Probability density')                   
plt.show()
fig.savefig('Histogram of YOB.jpg')