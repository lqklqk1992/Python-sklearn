from preprocess import transform
from preprocess import fill_missing
from sklearn.linear_model.logistic import LogisticRegression
from lr import logisticRegression
from sklearn.naive_bayes import *
from naive_bayes import NaiveBayes 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import svm
import time

def main():
    # load training data
    filename_train = './data/train.csv'
    train_dataset = transform(filename_train)

    X = train_dataset['data']
    y = train_dataset['target']

    # fill in missing data (optional)
    X_full = fill_missing(X, 'mode', False)

    X_full_train, X_full_test, y_train, y_test = train_test_split(X_full, y, test_size=0.25, random_state=0)

    ### use the logistic regression
    print('Train the logistic regression classifier')
    """ your code here """
    lr_model = LogisticRegression()
    start_time = time.time()
    lr_model.fit(X_full_train,y_train)
    elapsed_time = time.time() - start_time
    y_predict = lr_model.predict(X_full_test)
    print('The accuracy of the sklearn lr classifier: '+str(sum(y_test ==  y_predict)/y_test.shape[0])+' elapsed time: '+str(elapsed_time))
    clf = logisticRegression()
    start_time = time.time()
    clf.fit(X_full_train,y_train)
    elapsed_time = time.time() - start_time
    y_predict = clf.predict(X_full_test)
    print('The accuracy of my lr classifier: '+str(sum(y_test ==  y_predict)/y_test.shape[0])+' elapsed time: '+str(elapsed_time))
    
    ### use the naive bayes
    print('Train the naive bayes classifier')
    """ your code here """
    nb_model = MultinomialNB()
    start_time = time.time()
    nb_model.fit(X_full_train, y_train)
    elapsed_time = time.time() - start_time
    y_predict = nb_model.predict(X_full_test)
    print('The accuracy of the sklearn nb classifier: '+str(sum(y_test ==  y_predict)/y_test.shape[0])+' elapsed time: '+str(elapsed_time))
    clf = NaiveBayes()
    start_time = time.time()
    clf = clf.fit(X_full_train, y_train)
    elapsed_time = time.time() - start_time
    y_predict = clf.predict(X_full_test)
    print('The accuracy of my nb classifier: '+str(sum(y_test ==  y_predict)/y_test.shape[0])+' elapsed time: '+str(elapsed_time))

    ## use the svm
    print('Train the SVM classifier')
    """ your code here """
    svm_model = svm.SVC(kernel='linear', C=1).fit(X_full_train, y_train)
    print(('The accuracy of the sklearn SVM classifier: %f')%(svm_model.score(X_full_test, y_test)))                       

    ## use the random forest
    print('Train the random forest classifier')
    rf_model = RandomForestClassifier(n_estimators=500)
    rf_model.fit(X_full_train, y_train)
    print(('The accuracy of the sklearn random forest classifier: %f')%(rf_model.score(X_full_test, y_test))) 


    ## get predictions
    df = pd.read_csv('./data/test.csv')
    UserID=df.loc[:,'UserID'].as_matrix()
    df = df.drop('UserID', 1)
    X_predict=df.as_matrix()
    for n in range(df.shape[1]):
        if df.iloc[:,n].dtypes!=np.int64 and df.iloc[:,n].dtypes!=np.float64:
            g= pd.get_dummies(X_predict[:,n])
            i=0
            for e in list(g):
                X_predict[:,n][X_predict[:,n]==e]=i
                i=i+1 
    X_full_predict = fill_missing(X_predict, 'mode', False)
    
    y_predict = lr_model.predict(X_full_predict)
    fo = open("./predictions/lr_predictions.csv", "w")
    fo.write("UserID,Happy\n");
    for i in range(y_predict.shape[0]):
        fo.write(("%d,%d\n")%(UserID[i],y_predict[i]));
    fo.close()

    y_predict = nb_model.predict(X_full_predict)
    fo = open("./predictions/nb_predictions.csv", "w")
    fo.write("UserID,Happy\n");
    for i in range(y_predict.shape[0]):
        fo.write(("%d,%d\n")%(UserID[i],y_predict[i]));
    fo.close()

    y_predict = svm_model.predict(X_full_predict)
    fo = open("./predictions/svm_predictions.csv", "w")
    fo.write("UserID,Happy\n");
    for i in range(y_predict.shape[0]):
        fo.write(("%d,%d\n")%(UserID[i],y_predict[i]));
    fo.close()
    
    y_predict = rf_model.predict(X_full_predict)
    fo = open("./predictions/rf_predictions.csv", "w")
    fo.write("UserID,Happy\n");
    for i in range(y_predict.shape[0]):
        fo.write(("%d,%d\n")%(UserID[i],y_predict[i]));
    fo.close()

if __name__ == '__main__':
    main()
