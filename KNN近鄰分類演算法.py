from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

#尋找最近的k個點中，類別A及B的個數，將測試點判給較多的類群

cancer =load_breast_cancer()
#print(cancer.feature_names)
x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=77)


# 初始化KNN分類器，設置鄰居數量K和其他參數
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# 訓練KNN分類器
knn_classifier.fit(x_train, y_train)

# 使用測試數據進行預測
y_pred = knn_classifier.predict(x_test)

# 計算準確性
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)




