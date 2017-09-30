from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from pipeline import Pipeline

iris = load_iris()
clf = DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

parameters = {'max_features' : [None, 1, 2, 3],
              'min_impurity_decrease' : [0.0, 0.05, 0.10, 0.15]}

pipe = Pipeline(DecisionTreeClassifier, iris.data, iris.target,
                parameters = parameters)

def quickrun():
    pipe.grid_search()
    print 'grid searched score', pipe.score()
