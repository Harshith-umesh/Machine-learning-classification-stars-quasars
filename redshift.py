import pandas as pd
import numpy as np
import random as rnd


import seaborn as sns
import matplotlib.pyplot as plt

import inspect
#import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from collections import Counter	
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier




class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))  
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        """Predict class for X."""
        return [self._predict(inputs) for inputs in X]

    def debug(self, feature_names, class_names, show_details=True):
        """Print ASCII visualization of decision tree."""
        self.tree_.debug(feature_names, class_names, show_details)

    def _gini(self, y):
        
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def _best_split(self, X, y):
        
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )

                # The Gini impurity of a split is the weighted average of the Gini
                # impurity of the children.
                gini = (i * gini_left + (m - i) * gini_right) / m

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
        
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    
    def get_params(self, deep=True):
        
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out  
        
        
    def set_params(self, **params):
        
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self    
        
          
        

class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None        
        
        
        
def classifyWithDecisionTree ( trainingData, results, testData ,y_test):
 clf_tree =DecisionTreeClassifier(max_depth=7)
 clf_tree.fit(trainingData, results)
 #acc_dtree=round(clf_tree.score(trainingData, results) * 100, 2)
 #acc_d=round(clf_tree.score(testData, y_test) * 100, 2)
 return clf_tree.predict(testData)#,acc_dtree,acc_d        
        
        
        
        
if __name__== "__main__":

 data = pd.read_csv('cat1.csv')
 
  
 y=data["class"]
 X=data.drop(data.columns[[0]],axis=1)
 R=X.drop(["galex_objid","sdss_objid","class","pred"],axis=1)


 x_train2, X_test, y_train2, Y_test = train_test_split(R, y,test_size=0.2)
 
 

 print('original count ', Counter(y_train2))
 
 

 #sm = SMOTE(sampling_strategy='minority', random_state=7)
 #x_train2,y_train2 = sm.fit_sample(x_train, y_train)
 
 x_train22=pd.DataFrame(x_train2)
 y_train22=pd.DataFrame({"class": y_train2})
 
 
 print("Resampled dataset shape", Counter(y_train2))
 x_train22.columns = x_train2.columns  #####
 
  
 model = ExtraTreesClassifier()
 model.fit(x_train2,y_train2)
 features=model.feature_importances_ 
 #print("\nfeatures", features)
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
 
 x_train2=x_train2.to_numpy()
 y_train2=y_train2.to_numpy()
  
 l=[]
 c=0
 
 for i in np.nditer(features):
  if i< 0.03 :
   l.append(c)
   #print("x",i,c)
   #print(x_train2.shape)
  c=c+1
  #print(c) 
  
 
 x=0
 #print(l) 
 
 #print(x_train2) 
 for i in l:
  x_train2=np.delete(x_train2,np.s_[i-x],1)
  x=x+1   
   
 X_test=X_test.drop(X_test.columns[l], axis = 1)  
 
 print("\n")
  
 
 
 X_test=X_test.to_numpy()
 Y_test=Y_test.to_numpy()
 
 #prediction6 = classifyWithDecisionTree(x_train2, y_train2, X_test,Y_test)
 #print("\nDtree\n",acc_dtree,"\t",acc_d)




classifier = BaggingClassifier(DecisionTreeClassifier(max_depth=7),max_samples=0.7, max_features=0.7)
classifier.fit(x_train2, y_train2)
prediction6 = classifier.predict(X_test)


from sklearn.model_selection import cross_val_score
print("\nk fold for decision trees\n")
scores1 = cross_val_score(classifier, x_train2, y_train2, cv=10, scoring = "accuracy")

print("Scores:", scores1)
print("Mean:", scores1.mean())
print("Standard Deviation:", scores1.std())


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

print("\nconfusion matrix for decision trees\n")


result = confusion_matrix(Y_test, prediction6) 

print(result) 
print('Accuracy Score :',accuracy_score(Y_test, prediction6))
print ('Report : ')
print (classification_report(Y_test, prediction6) )

