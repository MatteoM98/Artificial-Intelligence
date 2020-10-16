import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# Load dataset
iris = load_iris()

print('Features: ' + ', '.join(iris.feature_names))
print('Classes: ' + ', '.join(iris.target_names))

# example of first data
print('----------First data example---------------')
for feature, value in zip(iris.feature_names,iris.data[0]):
    print('Feature: ', feature)
    print('Values: ', value)
print('Classes: ', iris.target[0]) # 0=setosa   1=versicolour   2=virginica
print('-------------------------')



# Before classifier's training, i remove a part of exemples from dataset
# for testing classifier accurancy later  ->  Testing data

#remove an example for each class from dataset, obtaining the Training Data
test_idx = [0, 50, 100]     #50 position of first setosa    100 position of first versicolour   and so on

# Training data
train_target = np.delete(iris.target, test_idx)     #all excluse these three
train_data = np.delete(iris.data, test_idx, axis=0) #same

# Testing data
testing_target = iris.target[test_idx]  #only these three
testing_data = iris.data[test_idx]      #same


# Train classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# Preview
preview = clf.predict(testing_data)  #one setosa, one versicolour, one virginica
l = preview.tolist()
l1 = [0, 1, 2]
if l == l1:
    print('Correct!')
    print(preview)
else :
    print('Error')


# Print decision tree
from six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file = dot_data,
                     class_names = iris.target_names,
                     filled = True,
                     rounded = True,
                     impurity = False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('iris.pdf')





