

import pandas as pd
import numpy as np
import random as rnd


import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from collections import Counter	
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier



def classifyWithDecisionTree ( trainingData, results, testData ,y_test):
 clf_tree = tree.DecisionTreeClassifier(max_depth=5)
 clf_tree.fit(trainingData, results)
 acc_dtree=round(clf_tree.score(trainingData, results) * 100, 2)
 acc_d=round(clf_tree.score(testData, y_test) * 100, 2)
 return clf_tree.predict(testData),acc_dtree,acc_d





if __name__== "__main__":

 data = pd.read_csv('cat4.csv')
 
  
 y=data["class"]
 X=data.drop(data.columns[[0]],axis=1)
 R=X.drop(["galex_objid","sdss_objid","class","pred","spectrometric_redshift"],axis=1)


 X_train, X_test, Y_train, Y_test = train_test_split(R, y,test_size=0.2)
 
 

 print('original count ', Counter(Y_train))
 
 

 sm = SMOTE(sampling_strategy='minority', random_state=7)
 x_train2,y_train2 = sm.fit_sample(X_train, Y_train)
 
 x_train22=pd.DataFrame(x_train2)
 y_train22=pd.DataFrame({"class": y_train2})
 
 
 print("Resampled dataset shape", Counter(y_train2))
 x_train22.columns = X_train.columns
 
  
 model = ExtraTreesClassifier()
 model.fit(x_train2,y_train2)
 features=model.feature_importances_ 
 feat_importances = pd.Series(model.feature_importances_, index=x_train22.columns)
 feat_importances.nlargest(33).plot(kind='barh')
 #plt.savefig("output5.png")
 #plt.show()
 
 print("\n")
 
 corrmat = x_train22.corr()
 top_corr_features = corrmat.index
 plt.figure(figsize=(50,50))
 g=sns.heatmap(x_train22[top_corr_features].corr(),annot=True,cmap="RdYlGn")
 g.set_yticklabels(g.get_yticklabels(), rotation=0)
 g.set_xticklabels(g.get_xticklabels(), rotation=90)
 #plt.savefig("output6.png")
 #plt.show()
 
 
  
 l=[]
 c=0
 print(c)
 for i in np.nditer(features):
  if i< 0.03 :
   l.append(c)
   print("x",i,c)
   print(x_train2.shape)
  c=c+1
  print(c) 
  
 
 x=0
 print(l) 
 
 print(x_train2) 
 for i in l:
  x_train2=np.delete(x_train2,np.s_[i-x],1)
  x=x+1   
   
 X_test=X_test.drop(X_test.columns[l], axis = 1)  
 #print(x_train2.shape) 
 #print(x_train2) 
 print("\n")
  
 
 
 [prediction6,acc_dtree,acc_d] = classifyWithDecisionTree(x_train2, y_train2, X_test,Y_test)


 
 print("\nDtree\n",acc_dtree,"\t",acc_d)

from sklearn.model_selection import cross_val_score
print("\nk fold for decision trees\n")

#clf_tree = tree.DecisionTreeClassifier()
classifier = BaggingClassifier(tree.DecisionTreeClassifier(),max_samples=0.7, max_features=0.7)
classifier.fit(x_train2, y_train2)
prediction7 = classifier.predict(X_test)
#scores1 = cross_val_score(clf_tree, x_train2, y_train2, cv=10, scoring = "accuracy")
'''scores1 = cross_val_score(classifier, x_train2, y_train2, cv=10, scoring = "accuracy")

print("Scores:", scores1)
print("Mean:", scores1.mean())
print("Standard Deviation:", scores1.std())
'''

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

print("\nconfusion matrix for decision trees\n")


results2 = confusion_matrix(Y_test, prediction7) 

print(results2) 
print('Accuracy Score :',accuracy_score(Y_test, prediction7))
print ('Report : ')
print (classification_report(Y_test, prediction7) )







