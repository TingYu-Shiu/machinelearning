import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #混淆矩陣
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

#enseble中的bagging方法，隨機抽樣形成一棵樹(放回，可能重複)，決策方式同決策樹

cancer =load_breast_cancer()
#print(cancer.feature_names)
x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=77)

# n_estimators:決策樹的數量
# Max_depth:樹能生長的最大深度
# Min_samples_split:至少要多少樣本才能進行切分
# Min_samples_leaf:最終的葉子結束上要有多少樣本

model = RandomForestClassifier(max_depth=6,n_estimators=5)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
num_correct_samples = accuracy_score(y_test,y_pred, normalize =False)

print(f'number of correct_samples:{num_correct_samples}')

print(f'accuracy:{accuracy}')



plt.figure(figsize=(20,15),dpi=300)
for i, x in enumerate(model.estimators_):
    plt.subplot(1, 5, i+1)
    plot_tree(x, filled=True) #filled=True套色
    plt.title("Decision tree trained on all the Breast Cancer features")
plt.show()


