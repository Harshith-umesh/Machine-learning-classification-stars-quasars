#quasi stellar radio source - quasar . its spectrum was redshifted , the wavelength of its light stretched out, so it should be travelling from very far away to acquire the redshift. light worth many galaxies of light from a very small area of space. ecretion disks formed.
#covariance matrix or multiilinear 

# datra visualization , outlier detection . covariance matrix
# attenuation or extinction is the gradual loss of flux intensity through a medium
# blue stars are much hotter10,000-35,000+ K than red stars and their energy output is found in the ultraviolet range 300nm and shorter.
# class 0 = star and class 1=quasars , pred = predicted
#SDSS- sloane digital sky survey , GALEX-gelexy evolution explorer 


#The shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m) . So Y.shape[0] is n . shape is a tuple that gives dimensions of the array

## Remove three columns as index base 
#df.drop(df.columns[[0, 4, 2]], axis = 1, inplace = True)   To actually edit the original DataFrame, the “inplace” parameter can be set to True, and there is no returned value.

#The sign of the covariance can be interpreted as whether the two variables increase together (positive) or decrease together (negative). The magnitude of the covariance is not easily interpreted. A covariance value of zero indicates that both variables are completely independent.

#Dichotomous variables are again categorical but only have 2 possible values usually 0 and 1. 

import pandas as pd
import numpy as np
import random as rnd


import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from collections import Counter	
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier



def classifyWithDecisionTree ( trainingData, results, testData ,y_test):
 clf_tree = tree.DecisionTreeClassifier(max_depth=5)
 clf_tree.fit(trainingData, results)
 acc_dtree=round(clf_tree.score(trainingData, results) * 100, 2)
 acc_d=round(clf_tree.score(testData, y_test) * 100, 2)
 return clf_tree.predict(testData),acc_dtree,acc_d

def classifyWithKNeighbors ( trainingData, results, testData,y_test ):
 clf_KNN = KNeighborsClassifier(n_neighbors=6)
 clf_KNN.fit(trainingData,results)
 acc_knn = round(clf_KNN.score(trainingData, results) * 100, 2)
 acc_k2=round(clf_KNN.score(testData, y_test) * 100, 2)
 return clf_KNN.predict(testData) , acc_knn, acc_k2

def classifyWithRandomForest ( trainingData, results, testData ,y_test):
 random_forest = RandomForestClassifier(n_estimators=100)
 random_forest.fit(trainingData, results)
 acc_random_forest = round(random_forest.score(trainingData, results) * 100, 2)
 acc_a=round(random_forest.score(testData, y_test) * 100, 2)
 return random_forest.predict(testData) , acc_random_forest, acc_a


if __name__== "__main__":

 data = pd.read_csv('cat3.csv')
 print(data.shape)
 data1=data.drop(data.columns[[0]],axis=1)
 data1=data1.drop(["galex_objid","sdss_objid","pred","class","spectrometric_redshift"],axis=1)
 
 
  
 y=data["class"]
 X=data.drop(data.columns[[0]],axis=1)
 R=X.drop(["galex_objid","sdss_objid","class","pred","spectrometric_redshift"],axis=1)

 print("hi")
 X_train, X_test, Y_train, Y_test = train_test_split(R, y,test_size=0.2)
 
 import matplotlib.pyplot as plt
 #X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")].index.values].hist(figsize=[11,11])
 
 #plt.show()
 
 Y_train.hist(figsize=[11,11])
 #Y_train.plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')
 plt.title('stars=0 quasars=1')
 plt.ylabel('Counts')
                   
 plt.show()
 
