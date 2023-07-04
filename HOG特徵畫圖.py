import cv2
import numpy as np
from sklearn import svm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features

positive_image = cv2.imread('cat_resized.jpg')
positive_hog_features = extract_hog_features(positive_image)


positive_hog_features = positive_hog_features.reshape(-1,9)


# 取出第一個方向的直方圖
histogram = positive_hog_features[2]

# 將直方圖歸一化到 [0, 255] 的範圍
histogram = cv2.normalize(histogram, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# 將直方圖轉換為整數型別
histogram = histogram.astype(np.uint8)
histogram = histogram.flatten()

# 視覺化直方圖
plt.bar(range(len(histogram)), histogram)
plt.show()