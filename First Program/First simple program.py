from sklearn import tree

# Collect data

# 0 = bumpy
# 1 = smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# 0 = apple
# 1 = orange
labels = ["apple", "apple", "orange", "orange"]

# Train classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)     #for each set of data we define a specify class

# Make prediction
weight = int(input('Please enter a weight: '))
surface = int(input('Please enter 0 = bumpy or 1 = smooth: '))
prediction = clf.predict([[weight, surface]])

print(''.join(prediction).capitalize())



