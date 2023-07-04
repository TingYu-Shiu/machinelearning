import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#建立一個一個的弱分類器(針對答案的殘差在進行切分，一輪一輪逼近最小殘差(梯度下降)


cancer =load_breast_cancer()
x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=87)

#n_eatimators:弱分類器的數量(跌代次數)
gb = GradientBoostingClassifier(n_estimators=500, random_state=0)
gb.fit(x_train,y_train)
y_pred = gb.predict(x_test)
print(accuracy_score(y_test,y_pred))