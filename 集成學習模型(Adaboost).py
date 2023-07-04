import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#建立一個一個的弱分類器(將答錯特徵權重增強，答對降低，並算出此分類器權重(錯誤高，權重小)，然後換下一個分類器，最後進行線性累加)


cancer =load_breast_cancer()
x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=87)

#n_eatimators:弱分類器的數量(跌代次數)
clf = AdaBoostClassifier(n_estimators=500, random_state=0)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test,y_pred))