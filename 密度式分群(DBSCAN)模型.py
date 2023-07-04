import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from itertools import cycle,islice
from sklearn.cluster import MeanShift
import scipy

#產生半徑為eps的圓，圓內各點持續往外直到boarderpoint，完全未接觸到的點即為noisy point

#設定2D樣本資料
n_samples =1500
random_state = 100

#生成同心圓資料點
noisy_circles = datasets.make_circles(n_samples=n_samples,factor=0.5,noise=0.05)
#factor(內圈跟外圈距離)，noise(資料點稀疏程度)


#生成稀疏三群資料點
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.5,2.5,0.5], random_state=random_state,centers=3,n_features=2)
#cluster_std(資料點離散程度)，預設是三群(共1500筆),特徵預設為2(二維))

#生成斜向三群資料點(使用轉換矩陣)
x, y =datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation =[[0.6,-0.6],[-0.4,0.8]]
x_aniso = np.dot(x,transformation)
aniso = (x_aniso,y)
normal =(x,y)

datasets = [(noisy_circles,{'n_clusters':2}),
            (varied, {'n_clusters':3}),
            (normal,{'n_clusters':3}),
            (aniso, {'n_clusters':3})]
'''
for i , (dataset,algo_params) in enumerate(datasets):
    x,y =dataset
    x = StandardScaler().fit_transform(x)
'''


model1 = cluster.DBSCAN(min_samples=5,eps=0.2)
model2 = cluster.DBSCAN(min_samples=5,eps=0.7)
model3 = cluster.DBSCAN(min_samples=10,eps=1)
model4 = cluster.DBSCAN(min_samples=5,eps=1)

y_pred1 = model1.fit_predict(noisy_circles[0])
y_pred2 = model2.fit_predict(x)
y_pred3 = model3.fit_predict(x_aniso)
y_pred4 = model4.fit_predict(varied[0])


#colors =np.array(list(islice(cycle(['#377eb8','#ff7f00','#4daf4a']),3)))

colors =np.array(['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00'])

plt.figure(figsize=(15,10),dpi=300)
plt.subplot(2,2,1)
plt.scatter(noisy_circles[0][:,0],noisy_circles[0][:,1],color=colors[y_pred1])
plt.subplot(2,2,2)
plt.scatter(x[:,0],x[:,1],color=colors[y_pred2])
plt.subplot(2,2,3)
plt.scatter(x_aniso[:,0],x_aniso[:,1],color=colors[y_pred3])
plt.subplot(2,2,4)
plt.scatter(varied[0][:,0],varied[0][:,1],color=colors[y_pred4])

