import numpy as np
import pandas as pa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #混淆矩陣
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB #連續型離散化分布
from sklearn.naive_bayes import BernoulliNB #只是(1,0)特徵
from sklearn.naive_bayes import MultinomialNB 
#離散型,預設 > alpha=1.0 laplace_smoothing

#發生A後他是B or C 的機率，比較誰大判斷是誰

cancer =load_breast_cancer()
x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=1)

model =GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
num_correct_samples = accuracy_score(y_test,y_pred, normalize =False)

print(f'number of correct_samples:{num_correct_samples}')

print(f'accuracy:{accuracy}')



