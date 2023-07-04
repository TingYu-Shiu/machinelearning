import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# Set up training data
# (1) basic example:C=1
#labels = np.array([1, -1, -1, -1])
#trainingData = np.matrix([[501, 10], [255, 10], [501, 255], [10, 501]], dtype=np.float32)

#(2) 4-point example:C=1
#labels = np.array([1, 1, -1, -1])
#trainingData = np.array([[300, 300], [400, 301], [300, 500], [400, 499]], dtype=np.float32)

# (3) complicated example: please try to set C=0.00001 & C=1

labels = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
trainingData = np.array([[100, 100], [101, 200], [100, 300], [150, 200], [400, 50], [400, 100], [400, 200], [400,300], [400, 350], [500, 50], [500, 100], [500, 200], [500,300], [500, 350]], dtype=np.float32)


# 創建並訓練SVM模型
model = SVC(kernel='linear', probability=True, C=1)
model.fit(trainingData, labels)

plt.figure()

colors = np.array(['red', 'blue', 'black', 'yellow'])

# 繪製訓練數據點
for i in range(len(trainingData)):
    plt.scatter(trainingData[i][0], trainingData[i][1], color=colors[labels[i]])


# 超平面 w1 * x1 + w2 * x2 + b = 0
# 獲取超平面的權重和截距
w = model.coef_[0]
b = model.intercept_


# 定義繪圖範圍
x_range = np.linspace(min(trainingData[:, 0]), max(trainingData[:, 0]), 100)
y_range = np.linspace(min(trainingData[:, 1]), max(trainingData[:, 1]), 100)

# 創建網格點
X, Y = np.meshgrid(x_range, y_range)

# 計算超平面方程的值
Z = w[0] * X + w[1] * Y + b

# 繪製超平面
plt.contour(X, Y, Z, levels=[-1,0,1], colors='red')


# 繪製超平面兩側的填充區域
plt.contourf(X, Y, Z, levels=[-np.inf, -1, 0, 1, np.inf], colors=['blue', 'yellow', 'green'], alpha=0.3)


plt.show()