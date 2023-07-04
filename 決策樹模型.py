import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #混淆矩陣
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

#訊息滳entropy(-p*logp/或是 gini = 1-{(1/3)**2 + (2/3)**2}，用特徵切分，取增益值最大(總訊息商-分類後訊息商)，(亦即分類後純度最高(-p*log(p)分數最低))


cancer =load_breast_cancer()
#print(cancer.feature_names)
x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=1)

# Max_depth:樹能生長的最大深度
# Min_samples_split:至少要多少樣本才能進行切分
# Min_samples_leaf:最終的葉子結束上要有多少樣本

model = DecisionTreeClassifier(min_samples_leaf=3,max_depth=6,min_samples_split=10)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
num_correct_samples = accuracy_score(y_test,y_pred, normalize =False)

print(f'number of correct_samples:{num_correct_samples}')

print(f'accuracy:{accuracy}')

plt.figure(figsize=(20,15),dpi=300)
plot_tree(model, filled=True) #filled=True套色
plt.title("Decision tree trained on all the Breast Cancer features")
plt.show()
