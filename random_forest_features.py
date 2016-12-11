from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import load_boston
from sklearn import preprocessing
import numpy as np


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


# EXAMPLE DATASET.
# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']
X = preprocessing.scale(X)
names = boston["feature_names"]



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