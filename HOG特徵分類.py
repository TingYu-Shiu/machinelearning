import cv2
import numpy as np
from sklearn import svm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import os 
from sklearn.decomposition import PCA

# 載入數據集
# 假設你有一個包含正樣本和負樣本的數據集，並且它們已經被標記為正樣本和負樣本。

'''
cv2.COLOR_BGR2GRAY: 將BGR圖像轉換為灰度圖像。
cv2.COLOR_GRAY2BGR: 將灰度圖像轉換為BGR圖像。
cv2.COLOR_BGR2RGB: 將BGR圖像轉換為RGB圖像。
cv2.COLOR_RGB2BGR: 將RGB圖像轉換為BGR圖像。
cv2.COLOR_BGR2HSV: 將BGR圖像轉換為HSV（色相、飽和度、明度）色彩空間。
cv2.COLOR_HSV2BGR: 將HSV圖像轉換為BGR圖像。
'''

p_path ='./heavy_makeup_CelebA/train/heavy_makeup'
p_file_list = os.listdir(p_path)

n_path='./heavy_makeup_CelebA/train/no_heavy_makeup'
n_file_list = os.listdir(n_path)


ptest_path ='./heavy_makeup_CelebA/val/heavy_makeup'
ptest_file_list = os.listdir(ptest_path)

ntest_path='./heavy_makeup_CelebA/val/no_heavy_makeup'
ntest_file_list = os.listdir(ntest_path)

# 提取HOG特徵
def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features

# 加載並提取HOG特徵
train_images = []
train_labels = []

# 加載正樣本圖像並提取HOG特徵
for i in p_file_list:
    positive_image_path = p_path+'/'+i
    positive_image = cv2.imread(positive_image_path)
    positive_hog_features = extract_hog_features(positive_image)
    train_images.append(positive_hog_features)
    train_labels.append(1)  # 正樣本標籤為1

# 加載負樣本圖像並提取HOG特徵
for j in n_file_list:
    negative_image_path = n_path+'/'+j
    negative_image = cv2.imread(negative_image_path)
    negative_hog_features =  extract_hog_features(negative_image)
    train_images.append(negative_hog_features)
    train_labels.append(0)  # 負樣本標籤為0

test_images = []
test_labels = []

# 加載正樣本圖像並提取HOG特徵
for k in ptest_file_list:
    positive_image_path = ptest_path+'/'+k
    positive_image = cv2.imread(positive_image_path)
    positive_hog_features = extract_hog_features(positive_image)
    test_images.append(positive_hog_features)
    test_labels.append(1)  # 正樣本標籤為1

# 加載負樣本圖像並提取HOG特徵
for l in ntest_file_list:
    negative_image_path = ntest_path+'/'+l
    negative_image = cv2.imread(negative_image_path)
    negative_hog_features =  extract_hog_features(negative_image)
    test_images.append(negative_hog_features)
    test_labels.append(0)  # 負樣本標籤為0

#kernel: linear,poly,rbf,sigmoid,precomputed


train_images = np.array(train_images)
test_images = np.array(test_images)



pca = PCA(n_components=0.8)
x_train_pca = pca.fit_transform(train_images)
x_test_pca = pca.transform(test_images)



classifier = svm.SVC(kernel='linear', probability=True, C=0.01)
classifier.fit(x_train_pca, train_labels)


# 在測試集上進行預測
predictions = classifier.predict(x_test_pca)

# 計算準確率
accuracy = np.mean(predictions == test_labels)
print("Accuracy:", accuracy)
