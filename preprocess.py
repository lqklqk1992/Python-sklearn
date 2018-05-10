import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import Imputer


def transform(filename):
    """ preprocess the training data"""
    """ your code here """
    df = pd.read_csv(filename)
    target=df.loc[:,'Happy'].as_matrix()
    df = df.drop('Happy', 1)
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
    return {'data':data,'target':target}

def fill_missing(X, strategy, isClassified):
    """
     @X: input matrix with missing data filled by nan
     @strategy: string, 'median', 'mean', 'mode'
     @isclassfied: boolean value, if isclassfied == true, then you need build a
     decision tree to classify users into different classes and use the
     median/mean/mode values of different classes to fill in the missing data;
     otherwise, just take the median/mean/most_frequent values of input data to
     fill in the missing data
    """
    if strategy=='mode':
        strategy='most_frequent'

    im=Imputer(missing_values='NaN', strategy=strategy, axis=0)

    if isClassified==False:
        X_full=im.fit_transform(X)	
    elif isClassified==True:
    	im1=Imputer(missing_values='NaN', strategy=strategy, axis=0)
    	x=np.copy(X).astype(np.float64)
    	im.fit_transform(x[:,[1,4]])
    	x[:,1][np.isnan(x[:,1])]=im.statistics_[0]
    	x[:,4][np.isnan(x[:,4])]=im.statistics_[1]

    	for i in range(7):
    		im.fit_transform(x[(x[:,1]==0)&(x[:,4]==i)])
    		im1.fit_transform(x[(x[:,1]==1)&(x[:,4]==i)])
    		for n in range(x.shape[1]):
    			x[:,n][(np.isnan(x[:,n]))&(x[:,1]==0)&(x[:,4]==i)]=im.statistics_[n]
    			x[:,n][np.isnan(x[:,n])&(x[:,1]==1)&(x[:,4]==i)]=im1.statistics_[n]
    	X_full=x
    return X_full
