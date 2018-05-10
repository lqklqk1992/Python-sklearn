import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 6

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

df = pd.read_csv('../data/train.csv')
data=df.loc[:,['Income','Happy']].as_matrix()
data=data[~pd.isnull(data[:,0])]
d1=data[:,1][data[:,0]=='under $25,000']
d2=data[:,1][data[:,0]=='$25,001 - $50,000']
d3=data[:,1][data[:,0]=='$50,000 - $74,999']
d4=data[:,1][data[:,0]=='$75,000 - $100,000']
d5=data[:,1][data[:,0]=='$100,001 - $150,000']
d6=data[:,1][data[:,0]=='over $150,000']

Hfractions = (sum(d1)/d1.shape[0], sum(d2)/d2.shape[0], sum(d3)/d3.shape[0], sum(d4)/d4.shape[0], sum(d5)/d5.shape[0],sum(d6)/d6.shape[0])
Ufractions = ((d1.shape[0]-sum(d1))/d1.shape[0], (d2.shape[0]-sum(d2))/d2.shape[0], (d3.shape[0]-sum(d3))/d3.shape[0], (d4.shape[0]-sum(d4))/d4.shape[0], (d5.shape[0]-sum(d5))/d5.shape[0],(d6.shape[0]-sum(d6))/d6.shape[0])

fig, ax = plt.subplots()

rects1 = ax.bar(ind, Hfractions, width, color='r')
rects2 = ax.bar(ind + width, Ufractions, width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('Fraction within a certain income level')
ax.set_title('Bar chart of income and happiness')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('under 25,000', '25,001-50,000', '50,000-74,999', '75,000-100,000', '100,001-150,000','over 150,000'),fontsize=6)

ax.legend((rects1[0], rects2[0]), ('Happy', 'Unhappy'))
plt.show()
fig.savefig('Bar chart of income and happiness.jpg')