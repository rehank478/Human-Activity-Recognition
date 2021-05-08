import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix

data = pd.read_csv("svm_features.csv")

training_set, test_set = train_test_split(data, test_size = 0.25, random_state = 3)

print(training_set)
X_train = training_set.iloc[:,0:14].values

Y_train = training_set.iloc[:,14].values

X_test = test_set.iloc[:,0:14].values
Y_test = test_set.iloc[:,14].values

classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)
test_set["Predictions"] = Y_pred

print(test_set)

cm = confusion_matrix(Y_test,Y_pred)
print(cm)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)