import numpy as np
import matplotlib.pyplot as plt # 需安裝 pillow 才能讀 JPEG
from matplotlib import image
from sklearn.cluster import MiniBatchKMeans,KMeans
# K 值 (要保留的顏色數量)
K = 10
# 讀取圖片
image = image.imread("cat_resized.jpg") / 255
w, h, d = tuple(image.shape)
image_data = np.reshape(image, (w * h, d))

# 將顏色分類為 K 種
kmeans = MiniBatchKMeans(n_clusters=K,init='k-means++')
labels = kmeans.fit_predict(image_data)
centers = kmeans.cluster_centers_
# 根據分類將顏色寫入新的影像陣列
image_compressed = np.zeros(image.shape)
label_idx = 0
for i in range(w):
    for j in range(h):
        image_compressed[i][j] = centers[labels[label_idx]]
        label_idx += 1
# 如果想儲存壓縮後的圖片, 將下面這句註解拿掉
#plt.imsave('./compressed.jpg', image_compressed)
# 顯示原圖跟壓縮圖的對照
plt.figure(figsize=(12, 9))
plt.subplot(211)
plt.title('Original photo')
plt.imshow(image)
plt.subplot(212)
plt.title(f'Compressed to KMeans={K} colors')
plt.imshow(image_compressed)
plt.tight_layout()
plt.show()