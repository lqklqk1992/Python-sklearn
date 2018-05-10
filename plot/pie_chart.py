import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../data/train.csv')
data=df.loc[:,['Gender','Happy']].as_matrix()
men=data[data[:,0]=='Male']
women=data[data[:,0]=='Female']

fig = plt.figure('Pie chart for the fraction of happy men/women')
fig.suptitle('Pie chart for the fraction of happy men/women', fontsize=14, fontweight='bold')
plt.subplot(121)
plt.pie(np.array([sum(men[:,1]==1),sum(men[:,1]==0)]),labels=['happy','unhappy'])
plt.xlabel('Men') 
plt.subplot(122)
plt.pie(np.array([sum(women[:,1]==1),sum(women[:,1]==0)]),labels=['happy','unhappy'])
plt.xlabel('Women')                 
plt.show()
fig.savefig('Pie chart for the fraction of happy.jpg')