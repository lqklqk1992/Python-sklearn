import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import parallel_coordinates

df = pd.read_csv('../data/train.csv')
data=df.loc[0:99,['Gender','Income','HouseholdStatus','EducationLevel','Party','Happy']] 
for n in range(5):
    g= pd.get_dummies(data.iloc[:,n])
    i=0
    for e in list(g):
    	data.iloc[:,n][data.iloc[:,n]==e]=i
    	i=i+1  

fig = plt.figure('Parallel Coordinates Plot')
parallel_coordinates(data, 'Happy',color=('r','b'))
plt.show()
fig.savefig('Parallel Coordinates Plot.jpg')