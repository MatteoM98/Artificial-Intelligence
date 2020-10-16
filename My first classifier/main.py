from sklearn import datasets
from scipy.spatial import distance

#function
def euc(a, b):
    return distance.euclidean(a, b)

#personalized version of KNN algorithm  (with k = 1)
class ScrappyKNN():
    def fit(self, x_train, y_train):    #x: data    y:label (classes)
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):          #x_test: data that need to be tested
        prediction = []
        for row in x_test:
            label = self.closest(row)
            prediction.append(label)
        return prediction

    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


#main
iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5)

# classifier
my_classifier = ScrappyKNN()

my_classifier.fit(x_train, y_train)

prediction = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, prediction))




