import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#類神經網路，激發器(想成帶入數學式中激發出答案)

cancer =load_breast_cancer()
#print(cancer.feature_names)
features = pd.DataFrame(cancer.data, columns=cancer.feature_names)
target = pd.DataFrame(cancer.target,columns=['target'])
cancer_data = pd.concat([features,target],axis=1)
cancer_data = cancer_data[['worst concave points','worst perimeter','target']]

x = cancer_data.loc[:,['worst concave points','worst perimeter']].values

y = cancer_data.loc[:,['target']].values.flatten()

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)

sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

pla = Perceptron().fit(x_train,y_train)
y_pred = pla.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
num_correct_samples = accuracy_score(y_test,y_pred, normalize =False)

print(f'number of correct_samples:{num_correct_samples}')

print(f'accuracy:{accuracy}')

print(f'coef:{pla.coef_}, intercept:{pla.intercept_}')

