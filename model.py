import pandas as pd
import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn import svm



wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42)

clf = svm.SVC(gamma =0.001,kernel="linear")
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

#lm = LinearRegression()
#lm.fit(X_train,y_train)

pickle.dump(clf,open('model.pkl','wb'))
