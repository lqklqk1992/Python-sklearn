import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../data/train.csv')
data=df.loc[:,['YOB','Income']].as_matrix()
data=data[~pd.isnull(data[:,0])]
data=data[~pd.isnull(data[:,1])]
data[:,1][data[:,1]=='$25,001 - $50,000']=37500
data[:,1][data[:,1]=='$50,000 - $74,999']=62500
data[:,1][data[:,1]=='$75,000 - $100,000']=87500
data[:,1][data[:,1]=='$100,001 - $150,000']=125000
data[:,1][data[:,1]=='under $25,000']=25000
data[:,1][data[:,1]=='over $150,000']=150000
fig = plt.figure('Scatter plot of YOB and income')
fig.suptitle('Scatter plot of YOB and income', fontsize=14, fontweight='bold')
plt.scatter(x=data[:,0],y=data[:,1], c='b', marker='o',alpha=0.05)
plt.ylabel('Income')
plt.xlabel('YOB')                 
plt.show()
fig.savefig('Scatter plot of YOB and income.jpg')