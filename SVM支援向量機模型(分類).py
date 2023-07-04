import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# 找出最大的兩個超平面距離，以容忍數據錯誤,若為線性不可分或複雜運kernel映射到高維度切分
#kernel: linear,poly,rbf,sigmoid,precomputed,Polynomial kernel

cancer =load_breast_cancer()
#print(cancer.feature_names)
x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=77)

model = SVC(kernel='linear',probability=True,C=2)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
num_correct_samples = accuracy_score(y_test,y_pred, normalize =False)

print(f'number of correct_samples:{num_correct_samples}')

print(f'accuracy:{accuracy}')

model.predict_proba(x_test) #方法的probability需為true才行



