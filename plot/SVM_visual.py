import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import sys
sys.path.insert(0, '../')
from preprocess import transform
from preprocess import fill_missing

# import some data to play with
df = pd.read_csv('../data/train.csv')
y=df.loc[:,'Happy'].as_matrix()
df=df.loc[:,['Income','EducationLevel']]
data=df.as_matrix()

n=df.columns.get_loc("Income")
data[:,n][data[:,n]=='under $25,000']=0
data[:,n][data[:,n]=='$25,001 - $50,000']=1
data[:,n][data[:,n]=='$50,000 - $74,999']=2
data[:,n][data[:,n]=='$75,000 - $100,000']=3
data[:,n][data[:,n]=='$100,001 - $150,000']=4
data[:,n][data[:,n]=='over $150,000']=5

n=df.columns.get_loc("EducationLevel")
data[:,n][data[:,n]=='Current K-12']=0
data[:,n][data[:,n]=='High School Diploma']=1
data[:,n][data[:,n]=='Current Undergraduate']=2
data[:,n][data[:,n]=="Associate's Degree"]=3
data[:,n][data[:,n]=="Bachelor's Degree"]=4
data[:,n][data[:,n]=="Master's Degree"]=5
data[:,n][data[:,n]=='Doctoral Degree']=6

X=data
X = fill_missing(X, 'mode', False)

h = .2  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.LinearSVC(C=C).fit(X, y)
#svc = svm.LinearSVC(C=C).fit(X, y)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))



Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig = plt.figure('Visualize SVM')
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y,cmap=plt.cm.coolwarm)
plt.xlabel('Income')
plt.ylabel('EducationLevel')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVC with linear kernel')
plt.show()
fig.savefig('Visualize SVM.jpg')