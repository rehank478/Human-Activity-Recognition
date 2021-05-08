from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd


data = pd.read_csv("moreSub_features1.csv")

training_set, test_set = train_test_split(data, test_size = 0.20, random_state = 0)

print(training_set)
X_train = training_set.iloc[:,0:12].values

Y_train = training_set.iloc[:,12].values

X_test = test_set.iloc[:,0:12].values
Y_test = test_set.iloc[:,12].values
# print (X_test,Y_test)
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train, Y_train)

Y_pred = KNN.predict(X_test)
print(Y_pred)
print(Y_test)

cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of KNN For The Given Dataset : ", accuracy)

DecTree = DecisionTreeClassifier()
DecTree.fit(X_train, Y_train)

Y_pred = DecTree.predict(X_test)
print(Y_pred)
print(Y_test)

cm2 = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm2.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Decision Tree For The Given Dataset : ", accuracy)

SVC = SVC(kernel='linear', random_state = 1)
SVC.fit(X_train,Y_train)

Y_pred = SVC.predict(X_test)
print(Y_pred)
print(Y_test)

cm3 = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm3.diagonal().sum())/len(Y_test)
print("\nAccuracy Of support vector machine for The Given Dataset : ", accuracy)