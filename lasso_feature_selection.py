import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from heapq import nlargest

from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import LassoLarsCV
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor



''' Explanation of names
0. CRIM: per capita crime rate by town 
1. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
2. INDUS: proportion of non-retail business acres per town 
3. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
4. NOX: nitric oxides concentration (parts per 10 million) 
5. RM: average number of rooms per dwelling 
6. AGE: proportion of owner-occupied units built prior to 1940 
7. DIS: weighted distances to five Boston employment centres 
8. RAD: index of accessibility to radial highways 
9. TAX: full-value property-tax rate per $10,000 
10. PTRATIO: pupil-teacher ratio by town 
11. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
12. LSTAT: % lower status of the population 
13. MEDV: Median value of owner-occupied homes in $1000's
'''



# Nice formating of the model.
def pretty_print_linear(coeff, indecies, names):
    #print(coeff)
    model = str(round(coeff[indecies[0]], 3)) + ' * ' + str(names[indecies[0]])
    for i in range(1, len(indecies)):
    	model = model + ' + ' + str(round(coeff[indecies[i]], 3)) + ' * ' + str(names[indecies[i]])
  
    return model



# EXAMPLE DATASET.

# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']
XX = X
X = preprocessing.scale(X)
names = boston["feature_names"]

# Total number of features.
tot_num_features = X.shape[1]
print('Total number of features \t{}\n'.format(tot_num_features))

print('Feature selection using Lasso.')
# L1 norm promotes sparsity of features.
clf = Lasso()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.4)
sfm.fit(X, y)
chosen_features = sfm.transform(X)
n_features = sfm.transform(X).shape[1]

# Establishing indecies of the choosen features.
indecies = np.zeros(n_features)
names_features = np.empty(n_features, dtype='S10')
index_count = 0
for i in range(tot_num_features):
	if (chosen_features[0][index_count] == X[0][i]):
		indecies[index_count] = i
		names_features[index_count] = names[i]
		index_count += 1

print(indecies)
# Fitting the model.
lassocv = Lasso()
lassocv.fit(X, y)
coef = lassocv.coef_


  
#print ("Lasso model: ", pretty_print_linear(coef, indecies, names))
print('\n\n')







print('Feature selection using Ridge Regression with built-in cross-validation.') 
# L2 norm promotes sparsity of features.
clf = Ridge()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.5)
sfm.fit(X, y)
chosen_features = sfm.transform(X)
n_features = sfm.transform(X).shape[1]

# Establishing indecies of the choosen features.
indecies = np.zeros(n_features)
names_features = np.empty(n_features, dtype='S10')
index_count = 0
for i in range(tot_num_features):
	if (chosen_features[0][index_count] == X[0][i]):
		indecies[index_count] = i
		names_features[index_count] = names[i]
		index_count += 1


print(indecies)
# Fitting the model.
ridge = Ridge()
ridge.fit(X, y)
coef = ridge.coef_

#print ("Ridge model: ", pretty_print_linear(coef, indecies, names))
print('\n\n')










print('Feature selection using Orthogonal Matching Pursuit model.') 
# Getting factors using Orthogonal matching pursuit. 
clf = OrthogonalMatchingPursuitCV()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=.5)
sfm.fit(X, y)
chosen_features = sfm.transform(X)
n_features = sfm.transform(X).shape[1]

# Establishing indecies of the choosen features.
indecies = np.zeros(n_features)
names_features = np.empty(n_features, dtype='S10')
index_count = 0
for i in range(tot_num_features):
	if (chosen_features[0][index_count] == X[0][i]):
		indecies[index_count] = i
		names_features[index_count] = names[i]
		index_count += 1

print(indecies)
# Fitting the model.
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=names_features.size)
omp.fit(X, y)
coef = omp.coef_

#print ("OMP model: ", pretty_print_linear(coef, indecies, names))
print('\n\n')




''' Changed all X and y's to integers.
'''
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=None, step=1)
X = X.astype(int)
y = y.astype(int)
rfe.fit(X, y)
coef = rfe.support_
#coef = rfe.param_
print(coef)





# Random forest
forest = ExtraTreesRegressor(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print(indices)

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


rf = RandomForestRegressor()
rf.fit(X, y)
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True))

print(rf.feature_importances_[1])











