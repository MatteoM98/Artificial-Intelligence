from sklearn import datasets
from scipy.spatial import distance
from heapq import heappop, heappush, heapify
from collections import Counter


#function
def euc(a, b):
    return distance.euclidean(a, b)


#personalized version of KNN algorithm  (with k = 1)
class ScrappyKNN():
    def fit(self, x_train, y_train):    #x: data    y:label (classes)
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, k):          #x_test: data that need to be tested
        prediction = []
        for row in x_test:
            label = self.closest(row, k)
            prediction.append(label)
        return prediction

    def closest(self, row, k):
        #create heap
        heap = []
        heapify(heap)

        #fill heap
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            heappush(heap, (dist, str(self.y_train[i])))

        #get first k mins
        mins = []
        for i in range(0, k):
            mins.append(heappop(heap)[1])

        #create counter
        count = Counter(mins)
        best_type = count.most_common(1)[0][0]

        return int(best_type)




#main
iris = datasets.load_iris()

x = iris.data       #features
y = iris.target     #classes

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

# classifier
my_classifier = ScrappyKNN()

my_classifier.fit(x_train, y_train)

k = int(input('Select a k value: '))

prediction = my_classifier.predict(x_test, k)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, prediction))




