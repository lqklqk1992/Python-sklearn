from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, '../')
from preprocess import transform
from preprocess import fill_missing

def biplot(score,coeff,pcax,pcay,labels=None):
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    n=score.shape[1]
    
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    scalepca1=1.0/(coeff[pca1].max()- coeff[pca1].min())
    scalepca2=1.0/(coeff[pca2].max()- coeff[pca2].min())
    vectorMultiply=20

    plt.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        x=(coeff-coeff.mean(1))[pca1,i]*scalepca1
        y=(coeff-coeff.mean(1))[pca2,i]*scalepca2
        plt.arrow(0, 0, x, y,color='r',alpha=0.5)
        x=(coeff-coeff.mean(1))[pca1,i]*scalepca1*1.15
        y=(coeff-coeff.mean(1))[pca2,i]*scalepca2*1.15
        if x<-1 or x>1 or y<-1 or y>1:
        	if abs(x)>abs(y):
        		x=(coeff-coeff.mean(1))[pca1,i]*scalepca1
        		y=(coeff-coeff.mean(1))[pca2,i]*scalepca2*(1.0/abs(x))
        		x=x*(1.0/abs(x))
        	else:
        		y=(coeff-coeff.mean(1))[pca2,i]*scalepca2
        		x=(coeff-coeff.mean(1))[pca1,i]*scalepca1*(1.0/abs(y))
        		y=y*(1.0/abs(y))

        if labels is None:
            plt.text(x, y, "Var"+str(i+1), color='y', ha='center', va='center')
        else:
            plt.text(x, y, labels[i], color='y', ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    plt.grid()

df = pd.read_csv('../data/train.csv')
df = df.drop('UserID', 1)
data=df.as_matrix()

n=df.columns.get_loc("Income")
data[:,n][data[:,n]=='under $25,000']=0
data[:,n][data[:,n]=='$25,001 - $50,000']=1
data[:,n][data[:,n]=='$50,000 - $74,999']=2
data[:,n][data[:,n]=='$75,000 - $100,000']=3
data[:,n][data[:,n]=='$100,001 - $150,000']=4
data[:,n][data[:,n]=='over $150,000']=5

n=df.columns.get_loc("HouseholdStatus")
data[:,n][data[:,n]=='Single (no kids)']=0
data[:,n][data[:,n]=='Single (w/kids)']=1
data[:,n][data[:,n]=='Married (no kids)']=2
data[:,n][data[:,n]=='Married (w/kids)']=3
data[:,n][data[:,n]=='Domestic Partners (no kids)']=4
data[:,n][data[:,n]=='Domestic Partners (w/kids)']=5

n=df.columns.get_loc("EducationLevel")
data[:,n][data[:,n]=='Current K-12']=0
data[:,n][data[:,n]=='High School Diploma']=1
data[:,n][data[:,n]=='Current Undergraduate']=2
data[:,n][data[:,n]=="Associate's Degree"]=3
data[:,n][data[:,n]=="Bachelor's Degree"]=4
data[:,n][data[:,n]=="Master's Degree"]=5
data[:,n][data[:,n]=='Doctoral Degree']=6

for n in range(df.shape[1]):
    if df.iloc[:,n].dtypes!=np.int64 and df.iloc[:,n].dtypes!=np.float64:
    	g= pd.get_dummies(data[:,n])
    	i=0
    	for e in list(g):
    		data[:,n][data[:,n]==e]=i
    		i=i+1 

X_full = fill_missing(data, 'mode', False)
mins = np.min(X_full, axis=0)
maxs = np.max(X_full, axis=0)
X_full=(X_full-np.mean(X_full,axis=0))/(maxs-mins)

pca = PCA()
X_reduced=pca.fit_transform(X_full)

fig = plt.figure('PCA and biplot')
biplot(X_reduced,pca.components_,1,2,list(df))
plt.show()
fig.savefig('PCA and biplot.jpg')