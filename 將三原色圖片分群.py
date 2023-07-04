import numpy as np
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from sklearn.cluster import KMeans
from copy import deepcopy

img1_rgb = image.imread('test_clustering1.jpg')
#img1_rgb = cv2.imread('test_clustering2.jpg') 

plt.imshow(img1_rgb)

w, h, d = tuple(img1_rgb.shape)
X = np.reshape(img1_rgb, (w * h,d))
#print(X.shape)

# Number of training data
#n = X.shape[0]
# Number of features in the data
#c = X.shape[1]

kmeans = KMeans(n_clusters=3,init='k-means++')
labels = kmeans.fit_predict(X)


colors =np.array(['orange', 'blue', 'green'])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(20,-60)
ax.scatter(X[:, 2], X[:, 1], X[:, 0], c =colors[labels])
ax.set_xlabel('blue')
ax.set_ylabel('gree')
ax.set_zlabel('red')


Y = deepcopy(X)

new_color_map = np.array([[255,0,0],[0,255,0],[0,0,255]])
Y[labels==0] = new_color_map[0,:]
Y[labels==1] = new_color_map[1,:]
Y[labels==2] = new_color_map[2,:]

Y = np.reshape(Y,(480,720,3))
plt.imshow(Y)