from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加載數據集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 將數據集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)

# 創建 SVM 分類器
svm = SVC(kernel='linear', decision_function_shape='ovr')  # 使用 "one-vs-rest" 策略

# 擬合訓練數據
svm.fit(X_train, y_train)

# 預測測試數據
y_pred = svm.predict(X_test)

# 計算分類準確率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)